"""Frame emission, stable free-slot allocation, and retirement on CUDA."""
from math import pi
import warp as wp
from warp.utils import array_scan
from .state import ParticleArrays
from .source import sample_source


def requested_births(config, frame) -> int:
    if not config.emission_start <= frame <= config.emission_end:
        return 0
    if config.mode == 'BURST':
        return config.burst_count if frame == config.emission_start else 0
    return config.particles_per_frame


@wp.kernel
def prepare(d: ParticleArrays, mask: wp.array(dtype=int), ids: wp.array(dtype=wp.int64), offset: wp.int64):
    i = wp.tid()
    mask[i] = 1-d.active[i]
    ids[i] = offset+wp.int64(i)


@wp.kernel
def births(d: ParticleArrays, mask: wp.array(dtype=int), prefix: wp.array(dtype=int),
           requested: wp.int64, offset: wp.int64, volume: float,
           positions: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
           faces: wp.array(dtype=int), bary: wp.array(dtype=wp.vec2),
           islands: wp.array(dtype=int), valid: wp.array(dtype=int),
           counts: wp.array(dtype=wp.int64), ledger: wp.array(dtype=wp.float64)):
    i = wp.tid()
    n = mask.shape[0]-1
    free_count = prefix[n]+mask[n]
    if i == 0:
        counts[0] += requested
        counts[2] += wp.max(wp.int64(0), requested-wp.int64(free_count))
    rank = prefix[i]
    if mask[i] == 1 and wp.int64(rank) < requested:
        if valid[rank] == 1:
            d.position[i] = positions[rank]
            d.velocity[i] = wp.vec3(0.0)
            d.normal[i] = normals[rank]
            d.face[i] = faces[rank]
            d.bary[i] = bary[rank]
            d.island[i] = islands[rank]
            d.ids[i] = offset+wp.int64(rank)
            d.path[i] = offset+wp.int64(rank)
            d.age[i] = 0.0
            d.volume[i] = volume
            d.active[i] = 1
            d.state[i] = 0
            d.limited[i] = 0
            wp.atomic_add(counts,1,wp.int64(1))
            wp.atomic_add(ledger,0,wp.float64(volume))
        else:
            wp.atomic_add(counts,3,wp.int64(1))


@wp.kernel
def retirement(d: ParticleArrays, lifetime: float, kill_height: float, ledger: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if d.active[i] == 1 and (d.age[i] >= lifetime or d.position[i][2] < kill_height):
        wp.atomic_add(ledger,1,wp.float64(d.volume[i]))
        d.active[i] = 0


def emit(pool, source, config, frame) -> None:
    emit_batch(pool,source,config,frame,requested_births(config,frame))


def emit_batch(pool, source, config, frame, requested: int) -> None:
    if isinstance(requested,bool) or not isinstance(requested,int) or not 0 <= requested <= 2**31-1:
        raise ValueError('Requested births must be a nonnegative int32 count')
    if requested == 0:
        return
    if pool.next_id > 2**63-1-pool.capacity:
        raise RuntimeError('Particle ID range exhausted; reset the simulation')
    wp.launch(prepare,pool.capacity,inputs=[pool.data,pool.mask,pool.candidate_ids,
              wp.int64(pool.next_id)],device=pool.device)
    array_scan(pool.mask,pool.prefix,inclusive=False)
    samples = sample_source(source,config.seed,frame,pool.candidate_ids,output=pool.samples,
                            count=min(requested,pool.capacity))
    wp.launch(births,pool.capacity,inputs=[pool.data,pool.mask,pool.prefix,wp.int64(requested),
        wp.int64(pool.next_id),4*pi/3*config.radius**3,samples.positions,samples.normals,
        samples.faces,samples.bary,samples.islands,samples.valid,pool.counters,pool.ledger],device=pool.device)
    pool.next_id += min(requested,pool.capacity)


def retire(pool, config) -> None:
    wp.launch(retirement,pool.capacity,inputs=[pool.data,config.lifetime,config.kill_height,pool.ledger],device=pool.device)
