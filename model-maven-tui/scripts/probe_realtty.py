import os,time,threading,tempfile,json
from winpty import PtyProcess
out=os.path.join(tempfile.gettempdir(),'mm-terminal-probe.json')
os.environ['MM_PROBE_OUT']=out
lua=os.path.abspath('model-maven-tui/scripts/probe_realtty.lua')
proc=PtyProcess.spawn([os.environ['MM_CI_NVIM'],'-u','NONE','-n','-c','luafile '+lua],dimensions=(45,180))
def drain():
    try:
        while True: proc.read(2048)
    except EOFError: pass
threading.Thread(target=drain,daemon=True).start()
for _ in range(200):
    if os.path.isfile(out): break
    time.sleep(.1)
if not os.path.isfile(out):
    proc.terminate(force=True)
    raise RuntimeError('Neovim did not produce a result')
result=json.load(open(out,encoding='utf-8'))
print(json.dumps(result,ensure_ascii=False))
if not any('HELLO_CONPTY' in line for line in result['lines']): raise RuntimeError('Terminal output absent')
