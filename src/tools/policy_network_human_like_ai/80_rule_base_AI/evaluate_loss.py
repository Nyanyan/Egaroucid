"""L21 disc loss of the frozen rule policy's Top-1 moves, on the full test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import numpy as np
from common import HERE, BASE, save, read, file_sha256
from calculate_quest_selected_move_loss import SelectedRunner
from calculate_quest_model_disc_loss import default_engine, engine_identity
from strength_engine import _create_windows_kill_job, _close_windows_job

KEY_DTYPE=np.dtype([('player','<u8'),('opponent','<u8'),('policy','i1')])
MISSING=-32768


def summarize(records, losses):
    if len(records)!=len(losses) or np.any(losses<0):
        raise ValueError('Missing or negative loss; never treat missing scores as zero')
    total=int(losses.sum(dtype=np.int64))
    if total!=sum(int(v) for v in losses):
        raise ValueError('Independent total differs')
    occupied=np.array([bin(int(p)|int(o)).count('1') for p,o in zip(records['player'],records['opponent'])])
    stages={}
    for name,mask in (('early_4_to_23',occupied<24),('middle_24_to_43',(occupied>=24)&(occupied<44)),('late_44_to_63',occupied>=44)):
        stages[name]=dict(positions=int(mask.sum()),mean_disc_loss=float(losses[mask].mean()) if mask.any() else None)
    return dict(positions=len(losses),total_disc_loss=total,mean_disc_loss=total/len(losses),
                zero_loss_positions=int(np.count_nonzero(losses==0)),stage=stages)


def run(args):
    import msvcrt
    import psutil
    root=args.input.resolve()
    out=args.output or root/'loss_l21'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+b')
    lock.seek(0)
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    proof=read(root/'agreement.json')
    if proof['status']!='complete':
        raise ValueError('Agreement experiment incomplete')
    for filename,key in (('selected_model.json','model_sha256'),('trained_rules_predictions.npy','predictions_sha256'),('test_records.npy','records_sha256')):
        if file_sha256(root/filename)!=proof[key]:
            raise ValueError('Frozen input changed: '+filename)
    records=np.load(root/'test_records.npy',allow_pickle=False)
    moves=np.load(root/'trained_rules_predictions.npy',allow_pickle=False)
    from policy_data import calc_legal_batch
    legal=calc_legal_batch(records['player'],records['opponent'])
    if np.any((legal&(np.uint64(1)<<moves.astype(np.uint64)))==0):
        raise ValueError('Illegal selected move')
    if args.max_positions:
        rng=np.random.default_rng(613)
        take=np.sort(rng.choice(len(records),min(args.max_positions,len(records)),replace=False))
        records,moves,legal=records[take],moves[take],legal[take]
    pairs=np.empty(len(records),dtype=KEY_DTYPE)
    pairs['player'],pairs['opponent'],pairs['policy']=records['player'],records['opponent'],moves
    keys,first,inverse=np.unique(pairs,return_index=True,return_inverse=True)
    unique_legal=legal[first]
    identity=dict(model_sha256=proof['model_sha256'],prediction_sha256=proof['predictions_sha256'],
                  source_records_sha256=proof['records_sha256'],positions=len(records),
                  unique_board_move_pairs=len(keys),pairs_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),
                  engine={**engine_identity(default_engine(),args.hash_level),'level':21,'clear_cache_per_position':False},
                  protocol='setboard relative player as Black; play selected move; analyze; warm persistent TT; book off',
                  loss_formula='max(played_score,best_other_move_score)-played_score; forced move => exact zero',
                  selection='argmax rule policy, ascending bit index ties',partial=bool(args.max_positions),
                  denominator='all original test positions, duplicates and forced moves included; passes absent',
                  implementation_sha256=file_sha256(Path(__file__)))
    if (out/'identity.json').exists():
        previous=read(out/'identity.json')
        if previous!=identity:
            old_semantics={k:v for k,v in previous.items() if k!='implementation_sha256'}
            new_semantics={k:v for k,v in identity.items() if k!='implementation_sha256'}
            if not args.resume_compatible_code or old_semantics!=new_semantics:
                raise ValueError('Loss resume identity differs')
            print('Resuming after an explicitly checked code change; score inputs and engine semantics are identical.',flush=True)
            identity=previous
        values=np.load(out/'scores.npy',mmap_mode='r+',allow_pickle=False)
    else:
        np.save(out/'pairs.npy',keys)
        np.save(out/'row_to_pair.npy',inverse)
        values=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='<i2',shape=(len(keys),3))
        values[:]=MISSING
        # One legal move has exactly zero regret regardless of its engine score.
        forced=(unique_legal&(unique_legal-np.uint64(1)))==0
        values[forced]=0
        values.flush()
        save(out/'identity.json',identity)
    if values.shape!=(len(keys),3):
        raise ValueError('Wrong saved score shape')
    complete=values[:,2]>=0
    if not np.all(values[complete,2]==np.maximum(values[complete,0],values[complete,1])-values[complete,0]):
        raise ValueError('Saved score formula differs')
    available=psutil.virtual_memory().available/2**30
    required=args.workers*1.4+4
    if available<required:
        raise RuntimeError(f'Insufficient free RAM: {available:.1f} GiB, require {required:.1f}')
    save(out/'preflight.json',dict(workers=args.workers,threads_per_engine=1,available_ram_gib=available,
                                  required_ram_gib=required,logical_cpus=psutil.cpu_count(),pid=os.getpid(),
                                  model_and_saved_moves_verified=True,identity_verified=True))
    history_path=out/'execution_runs.json'
    history=read(history_path) if history_path.exists() else []
    history.append(dict(pid=os.getpid(),workers=args.workers,threads_per_engine=1,
                        implementation_sha256=file_sha256(Path(__file__)),started_unix=time.time(),
                        resume_completed_pairs=int(complete.sum()),explicit_compatible_code=args.resume_compatible_code))
    save(history_path,history)
    todo,results,stop=queue.Queue(),queue.Queue(),threading.Event()
    # Shuffle to avoid concentrating the slowest game phase at the end.
    indices=np.flatnonzero(~complete)
    np.random.default_rng(613).shuffle(indices)
    for i in indices:
        todo.put(int(i))
    initial=done=int(complete.sum())
    started=last=time.monotonic()
    process_ids=set()
    pid_lock=threading.Lock()
    class Runner(SelectedRunner):
        def command(self):
            command=super().command()
            i=command.index('-hash')
            command[i+1]=str(args.hash_level)
            return command
    def worker():
        runner=Runner(default_engine(),level=21,threads=1,timeout_sec=3600,persistent=True,cancel_event=stop)
        job=None
        try:
            runner._ensure_process()
            job=_create_windows_kill_job(runner.proc)
            if os.name=='nt' and job is None:
                raise RuntimeError('Engine kill-on-parent-exit job could not be attached')
            with pid_lock:
                process_ids.add(runner.proc.pid)
            while not stop.is_set():
                try:
                    i=todo.get_nowait()
                except queue.Empty:
                    return
                key=keys[i]
                score=runner.analyze_move(int(key['player']),int(key['opponent']),int(key['policy']))
                results.put((i,score,None))
        except BaseException:
            results.put((None,None,traceback.format_exc()))
            stop.set()
        finally:
            runner.close()
            _close_windows_job(job)
    def progress(status='running'):
        elapsed=time.monotonic()-started
        rate=(done-initial)/elapsed if elapsed else 0
        eta=(len(keys)-done)/rate if rate else None
        remaining='estimating' if eta is None else f'{int(eta)//3600:02d}:{int(eta)%3600//60:02d}:{int(eta)%60:02d}'
        processes=[]
        for pid in tuple(process_ids):
            try:
                p=psutil.Process(pid)
                processes.append(dict(pid=pid,cpu_seconds=sum(p.cpu_times()[:2]),rss_gib=p.memory_info().rss/2**30))
            except psutil.NoSuchProcess:
                pass
        status_row=dict(status=status,pid=os.getpid(),completed_pairs=done,total_pairs=len(keys),original_test_positions=len(records),
                       workers=args.workers,engine_processes=processes,elapsed_sec=elapsed,pairs_per_sec=rate,
                       eta_sec=eta,updated_unix=time.time(),available_ram_gib=psutil.virtual_memory().available/2**30)
        save(out/'status.json',status_row)
        print(f'{status} | rule Top1 L21 | {done:,}/{len(keys):,} | '
              f'elapsed {elapsed:.0f}s | {rate:.2f} pairs/s | ETA {remaining} | engines {len(processes)}/{args.workers}',flush=True)
        return status_row
    progress()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker) for _ in range(min(args.workers,todo.qsize()))]
            try:
                while done<len(keys):
                    try:
                        i,result,error=results.get(timeout=1)
                    except queue.Empty:
                        if all(f.done() for f in futures):
                            raise RuntimeError('All engines stopped before results completed')
                    else:
                        if error:
                            raise RuntimeError(error)
                        values[i,:2]=result[:2]
                        values[i,2]=result[2]
                        done+=1
                    if time.monotonic()-last>=5:
                        values.flush()
                        status=progress()
                        if done-initial>=args.workers and len(status['engine_processes'])>0:
                            prior=out/'runtime_verified.json'
                            if not prior.exists() or read(prior).get('pid')!=os.getpid():
                                save(out/'runtime_verified.json',status)
                        last=time.monotonic()
                for f in futures:
                    f.result()
            finally:
                stop.set()
                values.flush()
        losses=np.asarray(values[inverse,2]).copy()
        summary=summarize(records,losses)
        np.save(out/'per_position_loss.npy',losses)
        save(out/'result.json',dict(status='complete',identity=identity,metrics=summary,
             scores_sha256=file_sha256(out/'scores.npy'),per_position_loss_sha256=file_sha256(out/'per_position_loss.npy'),
             verification='all frozen moves legal; engine board/ply/color/played move checked; complete original denominator; independent scalar sum matches'))
        progress('complete')
        print(f'COMPLETE | mean L21 disc loss = {summary["mean_disc_loss"]:.8f} | n={len(records):,}',flush=True)
    except BaseException as error:
        save(out/'status.json',dict(status='failed',error=str(error),pid=os.getpid(),completed_pairs=done,total_pairs=len(keys)))
        raise
    finally:
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'output/quest_800-1000')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--hash-level',type=int,default=25)
    parser.add_argument('--max-positions',type=int,default=0)
    parser.add_argument('--resume-compatible-code',action='store_true',
                        help='Allow a reviewed code-only change; all model, board, engine and protocol identities must match')
    args=parser.parse_args()
    if not 1<=args.workers<=8 or not 0<=args.hash_level<=29 or args.max_positions<0:
        parser.error('Workers must be 1..8; check hash level and sample count')
    if args.max_positions and args.output is None:
        parser.error('Smoke evaluation requires a separate --output')
    run(args)


if __name__=='__main__':
    main()
