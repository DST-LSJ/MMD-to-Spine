"""批量将当前文件夹中的人物VMD转换为Spine 4.3 JSON。"""

# ===== 日常只需修改这里：单位为秒 =====
START_SECONDS = 0
DURATION_SECONDS = 300.0  # 超过动作末尾自动截断；更长动作可增大此值
ENABLE_SECONDARY_MOTION = True  # 头发、双马尾、裙摆
ENABLE_BLINK = True

from pathlib import Path
import argparse
import sys
from 文档.batch_export import run_batch
from 文档.console_ui import wait_for_exit


def run_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vmd',nargs='*',type=Path,help='可选指定VMD；默认扫描当前文件夹，不递归')
    parser.add_argument('--folder',type=Path,default=Path.cwd(),help='扫描文件夹，默认当前工作文件夹')
    parser.add_argument('--output-dir',type=Path,help='输出文件夹，默认扫描文件夹')
    parser.add_argument('--start','--start-seconds',type=float,default=START_SECONDS)
    parser.add_argument('--duration','--duration-seconds',type=float,default=DURATION_SECONDS)
    parser.add_argument('-o','--output',type=Path,help='单个输入的指定JSON输出路径')
    parser.add_argument('--animation')
    parser.add_argument('--force',action='store_true',help='忽略转换记录，重新生成')
    parser.add_argument('--no-wait',action='store_true',help='完成后立即退出，不显示10秒倒计时')
    parser.add_argument('--no-secondary',action='store_true')
    parser.add_argument('--no-blink',action='store_true')
    parser.add_argument('--no-debug',action='store_true')
    parser.add_argument('--draw-order',action='store_true',help='可选动态遮挡，默认关闭')
    args=parser.parse_args(argv)
    folder=args.folder.resolve()
    if not folder.is_dir():
        parser.error(f'文件夹不存在：{folder}')
    paths=args.vmd or sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.vmd')
    if args.output and len(paths)!=1:
        parser.error('--output只能用于单个VMD')
    if not paths:
        print('当前文件夹没有VMD。请放入人物动作VMD后再次运行。')
        return []
    try:
        result=run_batch(paths,output_dir=args.output_dir or folder,
            start_seconds=args.start,duration_seconds=args.duration,force=args.force,
            output_path=args.output,animation_name=args.animation,
            secondary=ENABLE_SECONDARY_MOTION and not args.no_secondary,
            blink=ENABLE_BLINK and not args.no_blink,write_report=not args.no_debug,
            draw_order=args.draw_order)
    except ValueError as exc:
        parser.error(str(exc))
    if result['failed']:
        raise SystemExit(1)
    return [Path(p) for p in result['written']]


if __name__=='__main__':
    status=0
    try:
        run_cli()
    except SystemExit as exc:
        status=exc.code
    except KeyboardInterrupt:
        print('\n已取消转换。')
        status=130
    except Exception as exc:
        print(f'\n转换失败：{exc}')
        status=1
    finally:
        if not any(arg in sys.argv[1:] for arg in ('--no-wait','--help','-h')):
            wait_for_exit(10)
    raise SystemExit(status)
