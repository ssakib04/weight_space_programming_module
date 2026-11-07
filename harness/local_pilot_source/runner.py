from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import model_info
from datetime import datetime
from zoneinfo import ZoneInfo

def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def append_json(path, obj):
    with open(path,'a') as f:f.write(json.dumps(obj,default=str)+'\n');f.flush()
def event(**kw):
    append_json(OUT_DIR+'/events.jsonl',{'time_dhaka':datetime.now(ZoneInfo('Asia/Dhaka')).isoformat(),**kw})
def check_runtime():
    if torch.cuda.device_count()!=2:raise RuntimeError('Requires two CUDA GPUs; no CPU synthesis fallback')
    for i in range(2):
        if torch.cuda.get_device_properties(i).total_memory<14*2**30:raise RuntimeError('GPU memory below two T4-class cards')
    print(subprocess.check_output(['nvidia-smi'],text=True),flush=True)

def run_model(spec):
    cache=Path(tempfile.gettempdir())/('wsp_disposable_hf_'+digest(spec['id'])[:10])
    # This function owns only this new private model-cache directory; no shared HF cache touched.
    event(event='DISK_PREFLIGHT',model=spec['id'],cache_parent=str(cache.parent),free_bytes=shutil.disk_usage(cache.parent).free,required_gib=spec['disk_gib'])
    if cache.exists():raise RuntimeError('Private model cache already exists; inspect previous run before cleanup')
    if shutil.disk_usage(cache.parent).free<spec['disk_gib']*2**30:
        event(event='SKIPPED',model=spec['id'],reason='insufficient_download_disk',required_gib=spec['disk_gib']);return
    cache.mkdir()
    model=tok=None
    try:
        info=model_info(spec['id'],revision=spec['revision'],token=False)
        if info.gated:raise RuntimeError('Model unexpectedly gated: no authentication attempted')
        tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],cache_dir=str(cache),token=False,trust_remote_code=False)
        # Chat-template preflight before downloading weights.
        if not tok.chat_template:raise RuntimeError('Missing official chat template')
        probe=tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':task_prompt('modmul','none')}],tokenize=True,add_generation_prompt=True)
        if isinstance(probe,dict) or hasattr(probe,'keys'):probe=probe['input_ids']
        if probe and isinstance(probe[0],list):probe=probe[0]
        if len(probe)<100:raise RuntimeError('Implausibly short tokenized prompt')
        event(event='TOKENIZER_READY',model=spec['id'],revision=info.sha,prompt_tokens=len(probe),chat_template=tok.chat_template,chat_template_sha256=digest(tok.chat_template))
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],cache_dir=str(cache),token=False,trust_remote_code=False,quantization_config=q,device_map='balanced',max_memory={0:'12GiB',1:'12GiB'},torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True)
        if any(str(v) in ('cpu','disk') for v in model.hf_device_map.values()):raise RuntimeError('CPU/disk offload rejected; insufficient GPU fit')
        model.eval()
        model.generation_config.save_pretrained(OUT_DIR+'/'+digest(spec['id'])[:10]+'_generation_config')
        event(event='MODEL_READY',model=spec['id'],revision=info.sha,device_map=model.hf_device_map,footprint_bytes=model.get_memory_footprint(),quantization=q.to_dict(),model_config=model.config.to_dict())
        for task,level,rep in PILOT_CASES:
            if time.time()-T_START>MAX_TOTAL_S:
                event(event='TIME_BUDGET_STOP',model=spec['id']);break
            tag=re.sub(r'[^A-Za-z0-9_.-]','_',spec['id'])+f'_{task}_{level}_r{rep}'
            raw_path=Path(OUT_DIR)/(tag+'.json')
            if raw_path.exists():raise RuntimeError('Existing response; refusing to overwrite '+str(raw_path))
            user=task_prompt(task,level)
            msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}]
            rendered=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt')
            if not torch.is_tensor(ids):ids=ids['input_ids']
            if ids.shape[1]+MAX_NEW_TOKENS>model.config.max_position_embeddings:raise RuntimeError('Context limit exceeded; no prompt truncation allowed')
            ids=ids.to(model.get_input_embeddings().weight.device);seed=1000+rep;set_seed(seed)
            start=time.perf_counter()
            for i in range(2):torch.cuda.reset_peak_memory_stats(i)
            try:
                with torch.inference_mode():
                    out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=MAX_NEW_TOKENS,max_time=MAX_GENERATION_S,pad_token_id=tok.eos_token_id,**SAMPLING)
                gen=out[0,ids.shape[1]:]; token_ids=gen.tolist(); text=tok.decode(gen,skip_special_tokens=True)
                eos=model.generation_config.eos_token_id or tok.eos_token_id
                eos=eos if isinstance(eos,list) else [eos]
                stop='end_turn' if token_ids and token_ids[-1] in eos else ('max_tokens' if len(token_ids)>=MAX_NEW_TOKENS else 'time_limit')
                meta={'in_tok':int(ids.shape[1]),'out_tok':len(token_ids),'stop_reason':stop,'reply_source':'local','cost_usd':0.0,'latency_s':time.perf_counter()-start}
                record={'model_id':spec['id'],'revision':info.sha,'task':task,'prompt_level':level,'rep':rep,'seed':seed,'sampling':SAMPLING,'max_new_tokens':MAX_NEW_TOKENS,'max_generation_s':MAX_GENERATION_S,'messages':msgs,'rendered_prompt':rendered,'prompt_sha256':digest(rendered),'chat_template_sha256':digest(tok.chat_template),'response':text,'response_sha256':digest(text),'generated_token_ids':token_ids,'meta':meta,'peak_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(2)]}
                raw_path.write_text(json.dumps(record,indent=2)) # preserve raw BEFORE evaluation
                row={'task':task,'instance':'B','synth':spec['id'],'revision':info.sha,'prompt':level,'rep':rep,**meta,**process_reply(task,text,meta,tag)}
                row['incomplete']=stop!='end_turn'
                PER_RUN.append(row);save_all()
                print(tag,'valid',row['ok'],'accuracy',row['all_acc'],'stop',stop,flush=True)
                del out,gen
                if stop=='time_limit':
                    event(event='SLOW_GENERATION_STOP',model=spec['id'],tag=tag);break
            except Exception as e:
                event(event='GENERATION_ERROR',model=spec['id'],task=task,rep=rep,error=repr(e),traceback=traceback.format_exc());break
            finally:
                del ids;gc.collect();torch.cuda.empty_cache()
    except Exception as e:
        event(event='MODEL_ERROR',model=spec['id'],error=repr(e),traceback=traceback.format_exc())
        print('MODEL_ERROR',spec['id'],repr(e),flush=True)
    finally:
        del model,tok;gc.collect();torch.cuda.empty_cache()
        shutil.rmtree(cache) # only the exact disposable directory created above

check_runtime()
manifest={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','bitsandbytes','mistral-common','numpy']},'models':MODELS,'planned_cases':PILOT_CASES,'max_new_tokens':MAX_NEW_TOKENS,'max_generation_s':MAX_GENERATION_S,'sampling':SAMPLING,'gpus':[torch.cuda.get_device_properties(i).name for i in range(2)],'note':'pilot only; no provider API usage; fixed permadd mapping; FP32 CPU evaluator'}
Path(OUT_DIR+'/manifest.json').write_text(json.dumps(manifest,indent=2))
for spec in MODELS:
    if time.time()-T_START>MAX_TOTAL_S:break
    run_model(spec)
# Keep incomplete and infrastructure failures distinguishable; do not silently pool with completed outputs.
B=pd.DataFrame([r for r in PER_RUN if r.get('instance')=='B'])
if len(B):
    summary=B.groupby(['synth','task','prompt'],dropna=False).agg(n=('success','size'),valid=('ok','sum'),successes=('success','sum'),mean_acc=('all_acc','mean'),incomplete=('incomplete','sum'))
    summary.to_csv(OUT_DIR+'/pilot_summary.csv');print(summary)
print('Pilot finished. Inspect events.jsonl for missing/aborted attempts. Not a completed benchmark.',flush=True)
