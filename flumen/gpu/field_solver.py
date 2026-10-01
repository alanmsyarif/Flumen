"""Conservative anchor-to-field transfer with deterministic bounded reductions."""
import numpy as np
from dataclasses import dataclass
from math import ceil, sqrt, isfinite
from time import perf_counter
import warp as wp
from warp.utils import radix_sort_pairs
from .state import ParticleArrays
from .surface_chart import chart_anchor


@wp.kernel
def contribution_keys(d: ParticleArrays, triangles: wp.array(dtype=wp.vec3i), level: int,
                      source_faces: int, keys: wp.array(dtype=wp.int64), values: wp.array(dtype=int),
                      amounts: wp.array(dtype=wp.float64), momentum: wp.array(dtype=wp.vec3d),
                      unsupported: wp.array(dtype=wp.float64)):
    i = wp.tid()
    valid = d.active[i] == 1 and d.state[i] == 0
    valid = valid and d.face[i] >= 0 and d.face[i] < source_faces
    valid = valid and d.bary[i][0]>=-1.e-6 and d.bary[i][1]>=-1.e-6 and d.bary[i][0]+d.bary[i][1]<=1.000001
    nodes = wp.vec3i(-1)
    weights = wp.vec3(0.)
    if valid:
        child, weights = chart_anchor(d.face[i], d.bary[i], level)
        nodes = triangles[child]
    elif d.active[i] == 1 and d.state[i] == 0:
        wp.atomic_add(unsupported,0,wp.float64(d.volume[i]))
    stride = wp.int64(d.active.shape[0]*3)
    for k in range(3):
        index = 3*i+k
        node = nodes[k]
        if node < 0: node = 2147483647
        keys[index] = wp.int64(node)*stride+wp.int64(index)
        values[index] = index
        volume = wp.float64(d.volume[i])*wp.float64(weights[k])
        amounts[index] = volume
        momentum[index] = wp.vec3d(wp.float64(d.velocity[i][0])*volume,
                                 wp.float64(d.velocity[i][1])*volume,wp.float64(d.velocity[i][2])*volume)


@wp.kernel
def node_ranges(keys: wp.array(dtype=wp.int64), count: int,
                starts: wp.array(dtype=int), ends: wp.array(dtype=int)):
    i = wp.tid()
    node = int(keys[i]//wp.int64(count))
    if node >= starts.shape[0]: return
    if i == 0 or keys[i-1]//wp.int64(count) != wp.int64(node): starts[node] = i
    if i == count-1 or keys[i+1]//wp.int64(count) != wp.int64(node): ends[node] = i+1


@wp.kernel
def segment_partials(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=int),
                     amounts: wp.array(dtype=wp.float64), incoming: wp.array(dtype=wp.vec3d),
                     partial_volume: wp.array(dtype=wp.float64), partial_momentum: wp.array(dtype=wp.vec3d),
                     count: int):
    # Each fixed 256-entry chunk has <=256 runs. Only run starts walk a local run.
    i = wp.tid()
    node = keys[i]//wp.int64(count)
    partial_volume[i] = wp.float64(0.)
    partial_momentum[i] = wp.vec3d(0.)
    if i%256 != 0 and keys[i-1]//wp.int64(count) == node: return
    end = wp.min(count, (i//256+1)*256)
    total = wp.float64(0.); vector = wp.vec3d(0.)
    for j in range(i,end):
        if keys[j]//wp.int64(count) != node: break
        index = values[j]
        total += amounts[index]; vector += incoming[index]
    partial_volume[i] = total; partial_momentum[i] = vector


@wp.kernel
def reduce_nodes(starts: wp.array(dtype=int), ends: wp.array(dtype=int),
                 partial_volume: wp.array(dtype=wp.float64), partial_momentum: wp.array(dtype=wp.vec3d),
                 volume: wp.array(dtype=wp.float64), momentum: wp.array(dtype=wp.vec3d),
                 areas: wp.array(dtype=wp.float64), thickness: wp.array(dtype=float),
                 velocity: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    total = wp.float64(0.); vector = wp.vec3d(0.)
    start = starts[i]; end = ends[i]
    if start >= 0:
        total = partial_volume[start]; vector = partial_momentum[start]
        begin = (start//256+1)*256
        for j in range(begin,end,256):
            total += partial_volume[j]; vector += partial_momentum[j]
    volume[i] = total; momentum[i] = vector
    thickness[i] = float(total/areas[i])
    velocity[i] = wp.vec3(0.)
    if total > wp.float64(0.):
        velocity[i] = wp.vec3(float(vector[0]/total),float(vector[1]/total),float(vector[2]/total))


class FieldBuffers:
    def __init__(self, chart, device: str):
        self.chart, self.device = chart, device
        count = len(chart.vertices)
        self.volume = wp.zeros(count,dtype=wp.float64,device=device)
        self.momentum = wp.zeros(count,dtype=wp.vec3d,device=device)
        self.thickness = wp.zeros(count,dtype=float,device=device)
        self.velocity = wp.zeros(count,dtype=wp.vec3,device=device)
        self.wetness = wp.zeros(count,dtype=float,device=device)
        self.unsupported = wp.zeros(1,dtype=wp.float64,device=device)
        self.starts = wp.full(count,-1,dtype=int,device=device)
        self.ends = wp.zeros(count,dtype=int,device=device)
        self.laplacian = wp.zeros(count,dtype=float,device=device)
        self.pressure_gradient = wp.zeros(count,dtype=wp.vec3,device=device)
        self.capillary_gradient = wp.zeros(count,dtype=wp.vec3,device=device)
        self.triangle_gradient = wp.zeros(len(chart.triangles),dtype=wp.vec3,device=device)
        self.rhs = wp.zeros(count,dtype=wp.vec3,device=device)
        self.velocity_temp = wp.zeros(count,dtype=wp.vec3,device=device)
        self.metrics = wp.zeros(3,dtype=wp.float64,device=device)
        self.force_limited = wp.zeros(1,dtype=int,device=device)
        self.capacity = 0
        self.keys=self.values=self.amounts=self.incoming=self.partial_volume=self.partial_momentum=None

    def reserve(self, capacity):
        if self.capacity == capacity: return
        count = capacity*3
        self.keys = wp.empty(count*2,dtype=wp.int64,device=self.device)
        self.values = wp.empty(count*2,dtype=int,device=self.device)
        self.amounts = wp.empty(count,dtype=wp.float64,device=self.device)
        self.incoming = wp.empty(count,dtype=wp.vec3d,device=self.device)
        self.partial_volume = wp.empty(count,dtype=wp.float64,device=self.device)
        self.partial_momentum = wp.empty(count,dtype=wp.vec3d,device=self.device)
        self.capacity = capacity

    def close(self):
        for name in tuple(self.__dict__):
            if name not in ('device','capacity'): setattr(self,name,None)


def deposit_attached(pool, prepared, buffers: FieldBuffers) -> None:
    buffers.reserve(pool.capacity)
    buffers.starts.fill_(-1); buffers.ends.zero_(); buffers.unsupported.zero_()
    count = pool.capacity*3
    wp.launch(contribution_keys,pool.capacity,inputs=[pool.data,prepared.chart.triangles_gpu,
        prepared.chart.level,len(prepared.source.triangles_cpu),buffers.keys,buffers.values,
        buffers.amounts,buffers.incoming,buffers.unsupported],device=pool.device)
    radix_sort_pairs(buffers.keys,buffers.values,count)
    wp.launch(node_ranges,count,inputs=[buffers.keys,count,buffers.starts,buffers.ends],device=pool.device)
    wp.launch(segment_partials,count,inputs=[buffers.keys,buffers.values,buffers.amounts,buffers.incoming,
        buffers.partial_volume,buffers.partial_momentum,count],device=pool.device)
    wp.launch(reduce_nodes,len(prepared.chart.vertices),inputs=[buffers.starts,buffers.ends,
        buffers.partial_volume,buffers.partial_momentum,buffers.volume,buffers.momentum,
        prepared.chart.areas_gpu,buffers.thickness,buffers.velocity],device=pool.device)


@dataclass
class FieldStep:
    substeps: int = 0
    represented_volume: float = 0.
    unrepresented_volume: float = 0.
    limited_count: int = 0
    field_ms: float = 0.
    contact_ms: float = 0.
    aggregation_ms: float = 0.


@wp.kernel
def measure_field(height: wp.array(dtype=float), velocity: wp.array(dtype=wp.vec3),
                  volume: wp.array(dtype=wp.float64), metrics: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_max(metrics,0,wp.float64(height[i]))
    wp.atomic_max(metrics,1,wp.float64(wp.length(velocity[i])))
    wp.atomic_add(metrics,2,volume[i])


@wp.kernel
def laplacian_field(height: wp.array(dtype=float), areas: wp.array(dtype=wp.float64),
                    offsets: wp.array(dtype=int), neighbors: wp.array(dtype=int),
                    weights: wp.array(dtype=float), result: wp.array(dtype=float)):
    i = wp.tid(); value = float(0.)
    for k in range(offsets[i],offsets[i+1]):
        value += weights[k]*(height[neighbors[k]]-height[i])
    result[i] = value/float(areas[i])


@wp.kernel
def face_gradient(scalar: wp.array(dtype=float), triangles: wp.array(dtype=wp.vec3i),
                  gradients: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
    i = wp.tid(); tri = triangles[i]
    result[i] = gradients[3*i]*scalar[tri[0]]+gradients[3*i+1]*scalar[tri[1]]+gradients[3*i+2]*scalar[tri[2]]


@wp.kernel
def scatter_gradient(gradients: wp.array(dtype=wp.vec3), triangles: wp.array(dtype=wp.vec3i),
                     triangle_areas: wp.array(dtype=float), areas: wp.array(dtype=wp.float64),
                     result: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    for k in range(3):
        node = triangles[i][k]
        wp.atomic_add(result,node,gradients[i]*(triangle_areas[i]/(3.*float(areas[node]))))


@wp.kernel
def field_kick(velocity: wp.array(dtype=wp.vec3), height: wp.array(dtype=float),
               normals: wp.array(dtype=wp.vec3), pressure: wp.array(dtype=wp.vec3),
               capillary: wp.array(dtype=wp.vec3), gravity: wp.vec3, sigma: float,
               pressure_cap: float, capillary_cap: float, drag: float, dt: float,
               limited: wp.array(dtype=int)):
    i = wp.tid(); normal = normals[i]
    if height[i] <= 0.:
        velocity[i] = wp.vec3(0.)
        return
    acceleration = gravity-normal*wp.dot(gravity,normal)
    p = -wp.length(gravity)*pressure[i]
    c = (sigma/1000.)*capillary[i]
    if wp.length(p) > pressure_cap:
        p *= pressure_cap/wp.length(p); wp.atomic_add(limited,0,1)
    if wp.length(c) > capillary_cap:
        c *= capillary_cap/wp.length(c); wp.atomic_add(limited,0,1)
    acceleration += p+c
    acceleration -= normal*wp.dot(acceleration,normal)
    value = velocity[i]-normal*wp.dot(velocity[i],normal)
    decay = wp.exp(-drag*dt)
    coefficient = dt
    if drag*dt > 1.e-5: coefficient = (1.-decay)/drag
    velocity[i] = value*decay+acceleration*coefficient


@wp.kernel
def viscosity_jacobi(rhs: wp.array(dtype=wp.vec3), prior: wp.array(dtype=wp.vec3),
                     areas: wp.array(dtype=wp.float64), offsets: wp.array(dtype=int),
                     neighbors: wp.array(dtype=int), weights: wp.array(dtype=float),
                     normals: wp.array(dtype=wp.vec3), nu_dt: float, output: wp.array(dtype=wp.vec3)):
    i = wp.tid(); total = wp.vec3(0.); diagonal = float(0.)
    factor = nu_dt/float(areas[i])
    for k in range(offsets[i],offsets[i+1]):
        weight = weights[k]*factor
        total += prior[neighbors[k]]*weight; diagonal += weight
    value = (rhs[i]+total)/(1.+diagonal)
    normal = normals[i]
    output[i] = value-normal*wp.dot(value,normal)


def _gradient(chart, buffers, scalar, output):
    output.zero_()
    wp.launch(face_gradient,len(chart.triangles),inputs=[scalar,chart.triangles_gpu,
        chart.gradients_gpu,buffers.triangle_gradient],device=buffers.device)
    wp.launch(scatter_gradient,len(chart.triangles),inputs=[buffers.triangle_gradient,chart.triangles_gpu,
        chart.triangle_areas_gpu,chart.areas_gpu,output],device=buffers.device)


def evolve_field(prepared, buffers: FieldBuffers, config, dt: float) -> FieldStep:
    if not isfinite(dt) or dt < 0: raise ValueError('Field dt must be nonnegative and finite')
    if dt == 0: return FieldStep()
    start = perf_counter(); chart = prepared.chart
    buffers.metrics.zero_()
    wp.launch(measure_field,len(chart.vertices),inputs=[buffers.thickness,buffers.velocity,
        buffers.volume,buffers.metrics],device=buffers.device)
    height,speed,volume = buffers.metrics.numpy()
    wave = sqrt(max(0.,np.linalg.norm(config.gravity)*height))
    capillary_wave = sqrt(config.surface_tension/(1000.*chart.min_edge))
    needed = max(1,ceil(dt*(speed+wave+capillary_wave)/(.25*chart.min_edge)))
    if needed > 64:
        raise RuntimeError(f'Field stability requires {needed} steps, exceeding 64; refine time settings')
    buffers.force_limited.zero_()
    wp.launch(laplacian_field,len(chart.vertices),inputs=[buffers.thickness,chart.areas_gpu,
        chart.offsets_gpu,chart.neighbors_gpu,chart.neighbor_weights_gpu,buffers.laplacian],device=buffers.device)
    _gradient(chart,buffers,buffers.thickness,buffers.pressure_gradient)
    _gradient(chart,buffers,buffers.laplacian,buffers.capillary_gradient)
    interval = dt/needed
    for step in range(needed):
        wp.launch(field_kick,len(chart.vertices),inputs=[buffers.velocity,buffers.thickness,chart.normals_gpu,
            buffers.pressure_gradient,buffers.capillary_gradient,wp.vec3(*config.gravity),config.surface_tension,
            config.repulsion_acceleration,config.cohesion_acceleration,config.resistance+config.surface_damping,
            interval,buffers.force_limited],device=buffers.device)
        if config.field_viscosity > 0:
            wp.copy(buffers.rhs,buffers.velocity)
            for iteration in range(8):
                wp.launch(viscosity_jacobi,len(chart.vertices),inputs=[buffers.rhs,buffers.velocity,
                    chart.areas_gpu,chart.offsets_gpu,chart.neighbors_gpu,chart.neighbor_weights_gpu,chart.normals_gpu,
                    config.field_viscosity*interval,buffers.velocity_temp],device=buffers.device)
                buffers.velocity,buffers.velocity_temp = buffers.velocity_temp,buffers.velocity
    from .surface import wetness_step
    wp.launch(wetness_step,len(chart.vertices),inputs=[buffers.thickness,buffers.wetness,dt,
        config.wetness_deposit_rate,config.wetness_drying_rate],device=buffers.device)
    limited = int(buffers.force_limited.numpy()[0])
    unsupported = float(buffers.unsupported.numpy()[0])
    return FieldStep(needed,float(volume),unsupported,limited,(perf_counter()-start)*1000)


def sample_field(prepared, buffers: FieldBuffers, face: int, bary: tuple):
    if not 0 <= face < len(prepared.source.triangles_cpu): raise ValueError('Invalid global face')
    weights = np.array([bary[0],bary[1],1.-sum(bary)],dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < -1e-6).any(): raise ValueError('Invalid barycentric anchor')
    child = face
    for _ in range(prepared.chart.level):
        if weights[0]>=.5: corner=0; weights=weights*2-[1,0,0]
        elif weights[1]>=.5: corner=1; weights=weights*2-[0,1,0]
        elif weights[2]>=.5: corner=2; weights=weights*2-[0,0,1]
        else: corner=3; weights=np.array([1-2*weights[2],1-2*weights[0],1-2*weights[1]])
        child = child*4+corner
    weights=np.maximum(weights,0); weights/=weights.sum()
    nodes=prepared.chart.triangles[child]
    return float(weights@buffers.thickness.numpy()[nodes]),tuple(weights@buffers.velocity.numpy()[nodes])
