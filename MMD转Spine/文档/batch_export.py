"""Batch conversion with content-verified, successful-output-only caching."""
import hashlib
import json
import math
from pathlib import Path

from .runtime_assets import ROOT, DOC, ASSETS, load_profile
from .dance_export import convert
from .vmd_reader import read_vmd
from .console_ui import ConsoleProgress


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime_digest():
    # Include every runtime setting, template, source rig and texture. Exclude
    # reports, caches, tests and generated animations.
    paths = list(DOC.glob('*.py'))+list(DOC.glob('*.json'))
    paths += list(ASSETS.glob('*.json'))
    paths += [p for p in (ROOT/'Texture2D').rglob('*') if p.is_file()]
    h = hashlib.sha256(str(ROOT).encode('utf-8'))
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode('utf-8'))
        h.update(digest(p).encode('ascii'))
    return h.hexdigest()


def save_cache(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def run_batch(paths, *, output_dir, start_seconds, duration_seconds, force=False,
              output_path=None, animation_name=None, secondary=True, blink=True,
              write_report=True, draw_order=False):
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError('起始秒数必须为非负有限数')
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('持续秒数必须为正有限数')
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    cache_path = folder/'.mmd_spine_cache.json'
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
        if not isinstance(cache,dict) or cache.get('version') != 1:
            cache = {'version':1,'entries':{}}
        if not isinstance(cache.get('entries'),dict):
            cache['entries']={}
    except (OSError,ValueError):
        cache = {'version':1,'entries':{}}
    profile = load_profile()
    limb_aliases = {name for key,aliases in profile['aliases'].items()
                   if key.startswith(('left_','right_')) for name in aliases}
    runtime = None
    result = {'written':[], 'skipped':[], 'ignored':[], 'failed':[]}
    options = dict(start_seconds=start_seconds,duration_seconds=duration_seconds,
        secondary=secondary,blink=blink,write_report=write_report,draw_order=draw_order,
        animation_name=animation_name)
    sources=list(dict.fromkeys(Path(p).resolve() for p in paths))
    for index,source in enumerate(sources,1):
        print(f'\n[{index}/{len(sources)}] {source.name}',flush=True)
        try:
            motion = read_vmd(source)
            if not limb_aliases.intersection(motion.bone_tracks):
                print(f'忽略：{source.name}（没有可识别的人物肢体轨道，可能是相机/道具/表情动作）')
                result['ignored'].append(str(source)); continue
            start = round(start_seconds*30)
            if start >= motion.source_max_frame:
                print(f'忽略：{source.name}（截取起点已到达动作末尾）')
                result['ignored'].append(str(source)); continue
            end = min(start+max(1,round(duration_seconds*30)),motion.source_max_frame)
            label = f'通用骨架_{start/30:g}秒起_{(end-start)/30:g}秒'
            destination = (Path(output_path).resolve() if output_path else
                           folder/(source.stem+'_'+label+'_spine.json'))
            if destination == source:
                raise ValueError('输出路径不能覆盖输入VMD')
            if runtime is None:
                runtime = runtime_digest()
            fingerprint = hashlib.sha256(json.dumps(dict(source=digest(source),
                runtime=runtime,options=options),sort_keys=True).encode('utf-8')).hexdigest()
            key = str(destination)
            previous = cache['entries'].get(key,{})
            if (not force and previous.get('fingerprint') == fingerprint
                    and destination.is_file() and previous.get('output_sha256') == digest(destination)):
                print(f'跳过：{source.name}（输入、设置和输出均未变化）')
                result['skipped'].append(str(destination)); continue
            bar=ConsoleProgress(f'[{index}/{len(sources)}]')
            try:
                out,report = convert(source,folder,output_path=destination,progress=bar.update,**options)
            finally:
                bar.close()
            cache['entries'][key] = dict(fingerprint=fingerprint,source=str(source),
                output_sha256=digest(out),source_seconds=report['source_seconds'])
            save_cache(cache_path,cache)
            result['written'].append(str(out))
            print(f'完成：{out.name}（源{start/30:g}～{end/30:g}秒）')
        except Exception as exc:
            result['failed'].append({'source':str(source),'error':str(exc)})
            print(f'失败：{source.name}：{exc}')
    print(f"完成 {len(result['written'])}，跳过 {len(result['skipped'])}，忽略 {len(result['ignored'])}，失败 {len(result['failed'])}")
    return result
