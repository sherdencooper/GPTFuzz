from fastchat.model import load_model, get_conversation_template, add_model_args
import torch
import openai
import os

@torch.inference_mode()
def create_model(args, model_path):
    model, tokenizer = load_model(
        model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    return model, tokenizer


def create_model_and_tok(args, model_path):
    # Note that 'moderation' is only used for classification and cannot be used for generation 
    openai_model_list = ['gpt-3.5-turbo-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4', 'gpt-4-0301', 'moderation']
    open_sourced_model_list = ['lmsys/vicuna-7b-v1.3', 'lmsys/vicuna-33b-v1.3', 'meta-llama/Llama-2-7b-chat-hf', 'lmsys/vicuna-13b-v1.3', 'THUDM/chatglm2-6b', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf','baichuan-inc/Baichuan-13B-Chat']
    supported_model_list = openai_model_list + open_sourced_model_list
    if model_path not in supported_model_list:
        print("Please provide a valid model name in the list: {}".format(supported_model_list))
        exit()

    if model_path in openai_model_list:
        if args.openai_key == 'You must have an OpenAI key':
            print("Please provide your OpenAI key or choose an open-sourced model")
            exit()
        else:
            # openai.api_key = args.openai_key   # just for convenience
            openai.api_key = args.openai_key
            MODEL = model_path
            TOK = None
    
    elif model_path in open_sourced_model_list:
        MODEL, TOK = create_model(args, model_path)

    return MODEL, TOK


def prepare_model_and_tok(args):
    if type(args.model_path) == str:
        MODEL, TOK = create_model_and_tok(args, args.model_path)
    elif type(args.model_path) == list:
        MODEL, TOK = [], []
        for model_path in args.model_path:
            model, tok = create_model_and_tok(args, model_path)
            MODEL.append(model)
            TOK.append(tok)
    else:
        raise NotImplementedError

    return MODEL, TOK