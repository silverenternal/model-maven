import os,time,threading,tempfile,json
from winpty import PtyProcess
out=os.path.join(tempfile.gettempdir(),'mm-preview-result.json')
os.environ['MM_PROBE_OUT']=out
lua=os.path.abspath('model-maven-tui/scripts/verify_live_preview_windows.lua')
proc=PtyProcess.spawn([os.environ['MM_CI_NVIM'],'-u','NONE','-n','-c','luafile '+lua],dimensions=(45,180))
output=[]
def drain():
    try:
        while True: output.append(proc.read(2048))
    except EOFError: pass
threading.Thread(target=drain,daemon=True).start()
for _ in range(300):
    if os.path.isfile(out): break
    time.sleep(.1)
if not os.path.isfile(out):
    print(''.join(output)[-4000:])
    proc.terminate(force=True)
    raise RuntimeError('Neovim did not produce a result')
result=json.load(open(out,encoding='utf-8'))
print(json.dumps(result,ensure_ascii=False))
if not result['ok']: raise RuntimeError(result['error'])
