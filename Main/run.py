import json
from argparse import ArgumentParser
from LLM import LegalLLMPredictor
from retriver import Retriever
from rank import Reranker
import torch
from LawDataProcessor import LawDataProcessor  
import pandas as pd
from transformers import AutoTokenizer
import jieba
from accelerate import Accelerator 


def load_questions_from_file(file_path):
    df = pd.read_csv(file_path)
    questions = []
    for _, row in df.iterrows():
        question = row.get('question', '') 
        questions.append({'question': question})
    return questions

def load_reference_answers(reference_answers_path):
    try:
        with open(reference_answers_path, 'r', encoding='utf-8') as f:
            reference_answers = json.load(f)
        return reference_answers
    except FileNotFoundError:
        print(f"Error: Reference answers file {reference_answers_path} not found.")
        return {}

def extract_key_terms(question):
    key_terms = jieba.lcut(question)
    return [term for term in key_terms if len(term) > 1]

def process_batch(questions, retriever, reranker, llm, reference_answers, args, accelerator):
    results = {}
    for idx, entry in enumerate(questions):
        print(f"Processing question {idx + 1}/{len(questions)}")
        result = process_single_question(idx, entry, retriever, reranker, llm, reference_answers, args, accelerator)
        results[idx] = result
        torch.cuda.empty_cache()
    return results

def process_single_question(idx, entry, retriever, reranker, llm, reference_answers, args, accelerator):
    query = entry['question']
    
   
    key_terms = extract_key_terms(query)
    
    question_chunks = [query[i:i+50] for i in range(0, len(query), 50)] 
    chunked_question = ' '.join(question_chunks)

    
    retrieval_res = retriever.retrieval(query, methods=args.retrieval_methods.split(','), k=args.num_input_docs)
    rerank_res = reranker.rerank(retrieval_res, ' '.join(key_terms))  
    torch.cuda.empty_cache()  
   
    valid_docs = rerank_res
    if not valid_docs:
        answer = "没有找到合适的答案"
    else:
        
        context = '\n'.join(valid_docs)
        question = chunked_question  

        
        inputs = llm.tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)  # 设置 max_length 和 truncation
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

     
        device = accelerator.device  
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

      
        max_length = 1024 

        
        if accelerator.device == 'cuda':
            with torch.cuda.amp.autocast():  
                answer = llm.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        else:
            answer = llm.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

      
        answer = llm.tokenizer.decode(answer[0], skip_special_tokens=True).strip()

   
    correct_answer = None
    for ref_answer in reference_answers:
        if ref_answer.get("question") == query:  
            correct_answer = ref_answer.get("answer")
            break

   
    result = {
        "origin_prompt": f"\n{query}",
        "prediction": answer,
        "refr": correct_answer if correct_answer else ""
    }

    return result


def main():
    args = arg_parse()
    
    reference_answers = load_reference_answers(args.reference_answers_path)

    try:
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print(f"Error: Corpus file {args.corpus_path} not found.")
        return

   
    accelerator = Accelerator()

    # Initialize Retriever, Reranker, and LLM
    retriever = Retriever(emb_model_name_or_path=args.emb_model_name_or_path, corpus=corpus, device=accelerator.device)
    reranker = Reranker(rerank_model_name_or_path=args.rerank_model_name_or_path)
    llm = LegalLLMPredictor(llm_model_name_or_path=args.llm_model_name_or_path, device=accelerator.device)


    llm.model, retriever, reranker = accelerator.prepare(llm.model, retriever, reranker)

    # Load questions from CSV
    questions = load_questions_from_file(args.test_query_path)

    # Process questions in batches
    batch_size = args.batch_size
    results_1_1_all = {}
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        results_batch = process_batch(batch, retriever, reranker, llm, reference_answers, args, accelerator)
        results_1_1_all.update(results_batch)

    # Save the results to JSON file
    try:
        with open(args.output_file_1, 'w', encoding='utf-8') as f:
            json.dump(results_1_1_all, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {args.output_file_1}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
