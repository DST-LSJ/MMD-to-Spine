"""Dependency-free progress and bounded console exit prompt."""
import math
import os
import sys
import time


class ConsoleProgress:
    def __init__(self, label, stream=None):
        self.stream=stream or sys.stdout
        self.label=label
        self.terminal=self.stream.isatty()
        self.last_time=-1.
        self.last_stage=None
        self.last_bucket=-1
        self.width=0
        self.active=False

    def update(self, fraction, stage):
        percent=max(0,min(100,int(fraction*100)))
        now=time.monotonic()
        changed=stage!=self.last_stage
        if self.terminal:
            if not changed and percent<100 and now-self.last_time<.1:
                return
        elif not changed and percent//10==self.last_bucket:
            return
        filled=percent//5
        text=f'{self.label} [{"#"*filled}{"-"*(20-filled)}] {percent:3d}% {stage}'
        if self.terminal:
            print('\r'+text+' '*max(0,self.width-len(text)),end='',file=self.stream,flush=True)
            self.width=len(text)
        else:
            print(text,file=self.stream,flush=True)
        self.last_time=now;self.last_stage=stage;self.last_bucket=percent//10;self.active=True

    def close(self):
        if self.active and self.terminal:
            print(file=self.stream,flush=True)
        self.active=False


def wait_for_exit(seconds=10, *, stream=None, clock=None, sleep=None, key_pressed=None):
    stream=stream or sys.stdout
    if not sys.stdin.isatty() or not stream.isatty():
        return 'noninteractive'
    clock=clock or time.monotonic
    sleep=sleep or time.sleep
    restore=None
    if key_pressed is None:
        if os.name=='nt':
            import msvcrt
            def key_pressed():
                if msvcrt.kbhit():
                    msvcrt.getwch()
                    return True
                return False
        else:
            import select
            import termios
            import tty
            fd=sys.stdin.fileno();settings=termios.tcgetattr(fd)
            tty.setcbreak(fd)
            restore=lambda:termios.tcsetattr(fd,termios.TCSADRAIN,settings)
            def key_pressed():
                if select.select([sys.stdin],[],[],0)[0]:
                    os.read(fd,1)
                    return True
                return False
    deadline=clock()+seconds;previous=None;reason='timeout'
    try:
        while True:
            remaining=max(0,math.ceil(deadline-clock()))
            if remaining!=previous:
                print(f'\r按任意键退出，或 {remaining:2d} 秒后自动退出。',end='',file=stream,flush=True)
                previous=remaining
            if key_pressed():
                reason='key';break
            if remaining==0:
                break
            sleep(.05)
    except KeyboardInterrupt:
        reason='key'
    finally:
        if restore:
            restore()
        print(file=stream,flush=True)
    return reason
