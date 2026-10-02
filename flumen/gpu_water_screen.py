"""Screen-space water for the live particle preview (viewport only, no mesh).

Passes, all inside the POST_VIEW draw handler:
  1. sphere sprites -> linear eye depth in an offscreen R32F target (nearest surface wins);
  2. narrow-range filter, horizontal then vertical: smooths depth using only samples
     within `range` of the centre depth, so the surface gets smooth without bleeding edges
     (after Truong & Yuksel, "A Narrow-Range Filter for Screen-Space Fluid Rendering");
  3. composite: rebuild eye position and normal from filtered depth, shade, write depth so
     scene geometry still occludes the water.
"""
import gpu
from gpu_extras.batch import batch_for_shader

_SPRITE_VERT = '''
void main() {
    vec4 eye = view * vec4(xyzr.xyz, 1.0);
    center = eye.xyz;
    radius = xyzr.w * radius_scale;
    gl_Position = proj * eye;
    gl_PointSize = max(2.0, 2.0 * radius * proj[1][1] * viewport_height * 0.5 / max(gl_Position.w, 1e-6));
}'''
_SPRITE_FRAG = '''
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(c, c);
    if (d2 > 1.0) discard;
    vec3 eye = center + vec3(0.0, 0.0, radius * sqrt(1.0 - d2));
    /* Water hidden behind the wetted surface never enters the buffers. */
    float zs = texelFetch(surface_tex, ivec2(gl_FragCoord.xy), 0).r;
    if (zs > 0.0 && -eye.z > zs + radius) discard;
    vec4 clip = proj * vec4(eye, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    depth_out = vec4(-eye.z, 0.0, 0.0, 1.0);
}'''
_SURFACE_VERT = '''
void main() { vec4 eye = view * vec4(pos, 1.0); surface_eye = eye.xyz; gl_Position = proj * eye; }'''
_SURFACE_FRAG = '''
void main() {
    /* Same explicit depth convention as the sprites, so both passes compare consistently. */
    vec4 clip = proj * vec4(surface_eye, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    depth_out = vec4(-surface_eye.z, 0.0, 0.0, 1.0);
}'''
_QUAD_VERT = '''
void main() { gl_Position = vec4(pos, 0.0, 1.0); }'''
_FILTER_FRAG = '''
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float z0 = texelFetch(depth_tex, p, 0).r;
    /* Empty pixels and see-through pixels (water behind the object showing through gaps in the
       front sheet; the offscreen pass has no scene geometry to hide it) take the mean of their
       nearer close neighbours when most of them agree. */
    float nearer = 0.0; int count = 0;
    for (int i = -2; i <= 2; i++) {
        if (i == 0) continue;
        float z = texelFetch(depth_tex, p + direction * i, 0).r;
        if (z > 0.0 && (z0 <= 0.0 || z < z0 - depth_range)) { nearer += z; count++; }
    }
    if (count >= 3) z0 = nearer / float(count);
    if (z0 <= 0.0) { depth_out = vec4(0.0); return; }
    int r = int(clamp(world_size * proj_scale / z0, 1.0, 16.0));
    float sigma = max(float(r) * 0.5, 0.5);
    float sum = 0.0, weights = 0.0;
    for (int i = -16; i <= 16; i++) {
        if (abs(i) > r) continue;
        float z = i == 0 ? z0 : texelFetch(depth_tex, p + direction * i, 0).r;
        if (z <= 0.0 || abs(z - z0) > depth_range) continue;   /* other surfaces never bleed in */
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += w * z; weights += w;
    }
    depth_out = vec4(sum / max(weights, 1e-9), 0.0, 0.0, 1.0);
}'''
_COMPOSITE_FRAG = '''
vec3 eye_from(float z, ivec2 p) {
    vec2 ndc = (vec2(p) + 0.5) / size * 2.0 - 1.0;
    if (proj[3][3] > 0.5) return vec3((ndc - vec2(proj[3][0], proj[3][1])) / vec2(proj[0][0], proj[1][1]), -z);
    return vec3((ndc + vec2(proj[2][0], proj[2][1])) * z / vec2(proj[0][0], proj[1][1]), -z);
}
vec3 eye_at(ivec2 p) { return eye_from(texelFetch(depth_tex, p, 0).r, p); }
vec3 surface_at(ivec2 p) { return eye_from(texelFetch(surface_tex, p, 0).r, p); }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy - offset);
    float z = texelFetch(depth_tex, p, 0).r;
    if (z <= 0.0) discard;
    float zs = texelFetch(surface_tex, p, 0).r;
    /* A film a fraction of a millimetre thick follows the surface it wets: shade it with the
       surface's smooth geometry. Water away from the surface (drops) keeps its own normals. */
    bool film = zs > 0.0 && zs - z < film_reach;
    vec3 e = film ? surface_at(p) : eye_at(p);
    /* Depth test with the nearer of raw sprite and filtered depth, nudged forward: a film only
       a fraction of a millimetre thick must not lose against the surface it wets. */
    float raw = texelFetch(raw_tex, p, 0).r;
    float test_z = (film ? zs : (raw > 0.0 ? min(raw, z) : z)) - depth_bias;
    /* A 3-pixel stencil: residual filtered-depth noise is far below a pixel, but 1-pixel
       differences would turn it into speckled normals. */
    int s = film ? 1 : 3;
    vec3 dx1 = (film ? surface_at(p + ivec2(s, 0)) : eye_at(p + ivec2(s, 0))) - e;
    vec3 dx0 = e - (film ? surface_at(p - ivec2(s, 0)) : eye_at(p - ivec2(s, 0)));
    vec3 dy1 = (film ? surface_at(p + ivec2(0, s)) : eye_at(p + ivec2(0, s))) - e;
    vec3 dy0 = e - (film ? surface_at(p - ivec2(0, s)) : eye_at(p - ivec2(0, s)));
    vec3 dx = abs(dx0.z) < abs(dx1.z) ? dx0 : dx1;     /* one-sided at silhouettes */
    vec3 dy = abs(dy0.z) < abs(dy1.z) ? dy0 : dy1;
    vec3 n = normalize(cross(dx, dy));
    vec3 v = proj[3][3] > 0.5 ? vec3(0.0, 0.0, 1.0) : normalize(-e);
    if (dot(n, v) < 0.0) n = -n;
    vec3 l = normalize(vec3(0.4, 0.6, 0.7));
    float diffuse = 0.35 + 0.65 * max(dot(n, l), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    float specular = pow(max(dot(reflect(-l, n), v), 0.0), 80.0);
    vec3 rgb = water.rgb * diffuse * (1.0 - fresnel) + vec3(0.85, 0.9, 1.0) * fresnel + vec3(specular);
    color = vec4(rgb, clamp(water.a + fresnel * 0.5, 0.0, 1.0));
    vec4 clip = proj * vec4(e * (test_z / -e.z), 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
}'''


def _shader(vertex, fragment, inputs, outputs, constants, samplers=(), interface=None):
    info = gpu.types.GPUShaderCreateInfo()
    for slot, (kind, name) in enumerate(inputs): info.vertex_in(slot, kind, name)
    for kind, name in constants: info.push_constant(kind, name)
    for slot, name in enumerate(samplers): info.sampler(slot, 'FLOAT_2D', name)
    if interface is not None: info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', outputs)
    info.depth_write('ANY')
    info.vertex_source(vertex); info.fragment_source(fragment)
    return gpu.shader.create_from_info(info)


class ScreenWater:
    """Owns shaders and per-size offscreen targets; draws one published particle batch."""
    def __init__(self):
        interface = gpu.types.GPUStageInterfaceInfo('flumen_sprite')
        interface.smooth('VEC3', 'center'); interface.flat('FLOAT', 'radius')
        self.sprite = _shader(_SPRITE_VERT, _SPRITE_FRAG, [('VEC4', 'xyzr')], 'depth_out',
                              [('MAT4', 'view'), ('MAT4', 'proj'), ('FLOAT', 'radius_scale'), ('FLOAT', 'viewport_height')],
                              samplers=('surface_tex',), interface=interface)
        self.filter = _shader(_QUAD_VERT, _FILTER_FRAG, [('VEC2', 'pos')], 'depth_out',
                              [('IVEC2', 'direction'), ('FLOAT', 'world_size'), ('FLOAT', 'proj_scale'), ('FLOAT', 'depth_range')],
                              samplers=('depth_tex',))
        self.composite = _shader(_QUAD_VERT, _COMPOSITE_FRAG, [('VEC2', 'pos')], 'color',
                                 [('MAT4', 'proj'), ('VEC2', 'size'), ('VEC2', 'offset'), ('VEC4', 'water'),
                                  ('FLOAT', 'depth_bias'), ('FLOAT', 'film_reach')],
                                 samplers=('depth_tex', 'raw_tex', 'surface_tex'))
        quad = {'pos': ((-1, -1), (1, -1), (1, 1), (-1, 1))}
        self.quads = {name: batch_for_shader(getattr(self, name), 'TRI_FAN', quad) for name in ('filter', 'composite')}
        self.targets = {}

    def _target(self, width, height):
        key = (width, height)
        if key not in self.targets:
            if len(self.targets) >= 4: self.targets.clear()   # several viewport sizes at most
            colors = [gpu.types.GPUTexture(key, format='R32F') for _ in range(4)]   # raw, pass 1, pass 2, surface
            # Raw sprites and the surface each own a depth buffer; filter passes need none.
            depths = {0: gpu.types.GPUTexture(key, format='DEPTH_COMPONENT32F'), 3: gpu.types.GPUTexture(key, format='DEPTH_COMPONENT32F')}
            self.targets[key] = (colors, depths)
        colors, depths = self.targets[key]
        # Framebuffer objects belong to the current GPU context and are not shared between contexts;
        # a cached one silently lost its depth attachment, so build them per draw (textures stay cached).
        return colors, [gpu.types.GPUFrameBuffer(depth_slot=depths.get(i), color_slots=(c,)) for i, c in enumerate(colors)]

    @staticmethod
    def occluder(vertices, triangles):
        """Depth-only batch of the collision surface, so water hidden behind it never reaches
        the offscreen buffers (it would otherwise show through gaps in the front sheet)."""
        interface = gpu.types.GPUStageInterfaceInfo('flumen_surface'); interface.smooth('VEC3', 'surface_eye')
        shader = _shader(_SURFACE_VERT, _SURFACE_FRAG, [('VEC3', 'pos')], 'depth_out',
                         [('MAT4', 'view'), ('MAT4', 'proj')], interface=interface)
        return shader, batch_for_shader(shader, 'TRIS', {'pos': vertices}, indices=triangles)

    @staticmethod
    def batch(xyzr):
        fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id='xyzr', comp_type='F32', len=4, fetch_mode='FLOAT')
        buffer = gpu.types.GPUVertBuf(fmt, len(xyzr)); buffer.attr_fill('xyzr', xyzr)
        return gpu.types.GPUBatch(type='POINTS', buf=buffer)

    def draw(self, batch, settings, occluder=None):
        x, y, width, height = gpu.state.viewport_get()
        view, proj = gpu.matrix.get_model_view_matrix(), gpu.matrix.get_projection_matrix()
        (raw, first, second, surface), (fb_raw, fb_first, fb_second, fb_surface) = self._target(width, height)
        world = settings.water_smoothing; scale = proj[1][1]*height*.5
        with fb_surface.bind():   # surface depth: occludes hidden water, shades films
            gpu.state.viewport_set(0, 0, width, height)
            fb_surface.clear(color=(0., 0., 0., 0.), depth=1.)
            gpu.state.depth_test_set('LESS'); gpu.state.depth_mask_set(True); gpu.state.blend_set('NONE')
            # The viewport may leave back-face culling on; the surface must rasterize both sides.
            culling = gpu.state.face_culling_get() if hasattr(gpu.state, 'face_culling_get') else None
            gpu.state.face_culling_set('NONE')
            if occluder is not None:
                shader, mesh = occluder
                shader.bind(); shader.uniform_float('view', view); shader.uniform_float('proj', proj); mesh.draw(shader)
            if culling is not None: gpu.state.face_culling_set(culling)
        with fb_raw.bind():
            gpu.state.viewport_set(0, 0, width, height)
            fb_raw.clear(color=(0., 0., 0., 0.), depth=1.)
            gpu.state.depth_test_set('LESS'); gpu.state.depth_mask_set(True); gpu.state.blend_set('NONE')
            gpu.state.program_point_size_set(True)
            self.sprite.bind()
            self.sprite.uniform_float('view', view); self.sprite.uniform_float('proj', proj)
            self.sprite.uniform_float('radius_scale', settings.water_radius_scale)
            self.sprite.uniform_float('viewport_height', float(height))
            self.sprite.uniform_sampler('surface_tex', surface)
            batch.draw(self.sprite)
            gpu.state.program_point_size_set(False)
        gpu.state.depth_test_set('NONE')
        for source, target, direction in ((raw, fb_first, (1, 0)), (first, fb_second, (0, 1))):
            with target.bind():
                gpu.state.viewport_set(0, 0, width, height)
                self.filter.bind(); self.filter.uniform_sampler('depth_tex', source)
                self.filter.uniform_int('direction', direction)
                self.filter.uniform_float('world_size', world); self.filter.uniform_float('proj_scale', scale)
                self.filter.uniform_float('depth_range', max(2.*world, .003))
                self.quads['filter'].draw(self.filter)
        gpu.state.viewport_set(x, y, width, height)
        gpu.state.depth_test_set('LESS_EQUAL'); gpu.state.depth_mask_set(True); gpu.state.blend_set('ALPHA')
        self.composite.bind(); self.composite.uniform_sampler('depth_tex', second)
        self.composite.uniform_sampler('raw_tex', raw); self.composite.uniform_sampler('surface_tex', surface)
        self.composite.uniform_float('depth_bias', .25*world+1e-4)
        self.composite.uniform_float('film_reach', max(4.*settings.water_radius_scale*1e-4, .002))
        self.composite.uniform_float('proj', proj); self.composite.uniform_float('size', (float(width), float(height)))
        self.composite.uniform_float('offset', (float(x), float(y)))
        self.composite.uniform_float('water', tuple(settings.water_color))
        self.quads['composite'].draw(self.composite)
        gpu.state.blend_set('NONE')
