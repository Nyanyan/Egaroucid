"""Build, resume, verify and optionally install a complete egbk3 leaf rebuild."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).absolute().parents[3]
SOURCE = Path(__file__).with_name("recalculate_leaves.cpp")


def sha256(path):
    with Path(path).open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def engine_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/engine").glob("*.hpp")) + [SOURCE]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive_run(directory):
    lock = (directory / "repair.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"\0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError("Another repair is using this work directory")
    try:
        yield
    finally:
        lock.close()


def build(directory, digest):
    executable = directory / ("recalculate_leaves.exe" if os.name == "nt" else "recalculate_leaves")
    stamp = directory / "build.json"
    if executable.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get("engine_sha256") == digest and saved.get("executable_sha256") == sha256(executable):
            return executable
    if (directory / "leaves.checkpoint").exists():
        raise RuntimeError("Engine changed since checkpoint; use a new work directory")
    print("Building repair tool; ETA provisional 15s", flush=True)
    if os.name == "nt":
        vcvars = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / (
            "Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat")
        if not vcvars.exists():
            raise RuntimeError("Visual Studio 2022 C++ build tools are required on Windows")
        command_file = directory / "build.cmd"
        command_file.write_text(
            '@echo off\ncall "' + str(vcvars) + '" >nul\n'
            'cl /nologo /std:c++17 /Zc:__cplusplus /EHsc /O2 /MT /arch:AVX2 /utf-8 /DNDEBUG "'
            + str(SOURCE) + '" /Fo"' + str(directory / "recalculate_leaves.obj")
            + '" /Fe"' + str(executable) + '"\nexit /b %ERRORLEVEL%\n')
        command = ["cmd.exe", "/d", "/c", str(command_file)]
    else:
        command = ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", str(SOURCE), "-o", str(executable)]
    with (directory / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if result.returncode:
        raise RuntimeError("Build failed; see " + str(directory / "build.log"))
    stamp.write_text(json.dumps(dict(engine_sha256=digest, executable_sha256=sha256(executable)), indent=2))
    return executable


def run_logged(command, directory, name):
    try:
        import psutil
    except ImportError:
        psutil = None
    print("Starting " + name + "; console closure stops this run. Completed searches are checkpointed.", flush=True)
    with (directory / (name + ".log")).open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", cwd=ROOT)
        (directory / "process.json").write_text(json.dumps(dict(pid=process.pid, command=[str(x) for x in command]), indent=2))

        def consume():
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=consume)
        reader.start()
        monitored = psutil.Process(process.pid) if psutil else None
        if monitored:
            monitored.cpu_percent(None)
        started = time.monotonic()
        peak_cpu = 0.0
        peak_threads = 0
        try:
            while process.poll() is None:
                time.sleep(1)
                if monitored and int(time.monotonic() - started) % 10 == 0:
                    try:
                        cpu = monitored.cpu_percent(None)
                        threads = monitored.num_threads()
                        memory = monitored.memory_info().rss / 2**30
                        peak_cpu = max(peak_cpu, cpu)
                        peak_threads = max(peak_threads, threads)
                        line = f"process CPU={cpu:.0f}% threads={threads} RAM={memory:.2f}GiB\n"
                        print(line, end="", flush=True)
                        (directory / "monitor.json").write_text(json.dumps(dict(
                            pid=process.pid, cpu_percent=cpu, threads=threads, memory_gib=memory,
                            elapsed_seconds=time.monotonic() - started), indent=2))
                    except psutil.Error:
                        pass
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            reader.join()
        status = dict(exit_code=process.returncode, seconds=time.monotonic() - started,
                      peak_cpu_percent=peak_cpu, peak_threads=peak_threads)
        (directory / (name + "-status.json")).write_text(json.dumps(status, indent=2))
        if process.returncode:
            raise RuntimeError(name + " failed; checkpoint retained; see log")
        return status


def install(output, destination, directory, expected_hash):
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(output)
    current = sha256(destination) if destination.exists() else None
    if current not in (expected_hash, digest):
        raise RuntimeError("Destination changed during repair; refusing replacement: " + str(destination))
    if destination.exists():
        previous = sha256(destination)
        backup = directory / ("backup-" + previous + ".egbk3")
        if not backup.exists():
            shutil.copy2(destination, backup)
        if sha256(backup) != previous:
            raise RuntimeError("Backup verification failed")
    temporary = destination.with_name(destination.name + ".leaf-repair.tmp")
    shutil.copyfile(output, temporary)
    if sha256(temporary) != digest:
        raise RuntimeError("Installation copy verification failed")
    os.replace(temporary, destination)
    if sha256(destination) != digest:
        raise RuntimeError("Installed book hash mismatch")
    print("Installed and verified " + str(destination), flush=True)


def verify_checkpoint(output, checkpoint, report):
    if not report["complete"] or report["completed_searches"] != report["searches"]:
        raise RuntimeError("Calculation is incomplete")
    data = output.read_bytes()
    seen = set()
    with checkpoint.open() as journal:
        if not journal.readline().startswith("egaroucid-leaf-repair-v1 "):
            raise RuntimeError("Invalid checkpoint header")
        for line in journal:
            fields = [int(x) for x in line.split()]
            if len(fields) != 6:
                raise RuntimeError("Malformed checkpoint row")
            index, value, move, level, nodes, milliseconds = fields
            if index in seen or not 0 <= index < report["boards"]:
                raise RuntimeError("Duplicate or invalid checkpoint index")
            if not (-64 <= value <= 64 and 0 <= move < 64 and 1 <= level <= 60):
                raise RuntimeError("Invalid checkpoint result")
            p = 14 + index * 25 + 22
            if struct.unpack_from('bbb', data, p) != (value, move, level):
                raise RuntimeError("Saved leaf differs from completed search")
            seen.add(index)
    if len(seen) != report["searches"]:
        raise RuntimeError("Not every required search has a completed checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "bin/resources/book.egbk3")
    parser.add_argument("--resources", type=Path, default=ROOT / "bin/resources")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--hash-level", type=int, default=25)
    parser.add_argument("--level", type=int, default=0, help="0 preserves each leaf level; undefined levels use board level, then Lv5")
    parser.add_argument("--limit", type=int, default=0, help="Pilot: calculate at most N new leaves; do not install partial output")
    parser.add_argument("--install-to", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.work_dir = args.work_dir.absolute()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.work_dir):
        digest = engine_digest()
        executable = build(args.work_dir, digest)
        original = args.work_dir / "input.egbk3"
        manifest_path = args.work_dir / "manifest.json"
        resources = args.resources.absolute()
        identity = dict(input_sha256=sha256(original if original.exists() else args.input),
                        engine_sha256=digest, executable_sha256=sha256(executable),
                        eval_sha256=sha256(resources / "eval.egev2"),
                        move_ordering_eval_sha256=sha256(resources / "eval_move_ordering_end.egev"),
                        hash_sha256=sha256(resources / "hash" / f"hash{args.hash_level}.eghs"),
                        level=args.level, hash_level=args.hash_level)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != identity:
                raise RuntimeError("Input, engine, evaluation or settings changed; use a new work directory")
        else:
            shutil.copy2(args.input, original)
            if sha256(original) != identity["input_sha256"]:
                raise RuntimeError("Input snapshot verification failed")
            manifest_path.write_text(json.dumps(identity, indent=2))
        output = args.work_dir / "repaired.egbk3"
        # A resume must still refer to this input, or to our already installed output.
        live_hash = sha256(args.input)
        if live_hash != identity["input_sha256"] and (not output.exists() or live_hash != sha256(output)):
            raise RuntimeError("Input changed; immutable snapshot and checkpoint retained")
        plan_path = args.work_dir / "installation-plan.json"
        destinations = [str(path.absolute()) for path in args.install_to]
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if sorted(plan) != sorted(destinations):
                raise RuntimeError("Installation destinations changed; keep the original installation plan")
        else:
            plan = {path: sha256(path) if Path(path).exists() else None for path in destinations}
            if plan:
                plan_path.write_text(json.dumps(plan, indent=2))
        common = [executable, "--input", original, "--output", output, "--level", str(args.level)]
        if not output.exists():
            command = common + ["--mode", "repair", "--resources", resources,
                "--checkpoint", args.work_dir / "leaves.checkpoint", "--report", args.work_dir / "repair-report.json",
                "--threads", str(args.threads), "--hash-level", str(args.hash_level), "--limit", str(args.limit)]
            run_logged(command, args.work_dir, "repair")
        if not output.exists():
            print("Pilot finished. No destination was changed. Resume with the same work directory and --limit 0.", flush=True)
            return
        run_logged(common + ["--mode", "verify", "--report", args.work_dir / "validation.json"], args.work_dir, "verify")
        validation = json.loads((args.work_dir / "validation.json").read_text())
        verify_checkpoint(output, args.work_dir / "leaves.checkpoint", validation)
        result = dict(identity, output_sha256=sha256(output), output_bytes=output.stat().st_size,
                      installed_to=[str(x.absolute()) for x in args.install_to])
        for destination in args.install_to:
            install(output, destination, args.work_dir, plan[str(destination.absolute())])
        (args.work_dir / "result.json").write_text(json.dumps(result, indent=2))
        print("COMPLETE: all leaves rebuilt and all requested destinations verified.", flush=True)


if __name__ == "__main__":
    main()
