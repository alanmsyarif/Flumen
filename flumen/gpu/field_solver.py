"""Conservative anchor-to-field transfer with deterministic bounded reductions."""
import numpy as np
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
