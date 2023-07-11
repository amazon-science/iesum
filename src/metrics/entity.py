# Copyright <first-edit-year> Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import os
import logging
import json
import tqdm

import numpy as np
import spacy
    

ENTITY_REWRITE = {
    "'s": '',
    "’s": '',
    "'": "",
    'The ': '',
    'the ': '',
    'US': 'United States',
    'U.S.': 'United States',
    'EU': 'European Union',
    'F.B.I': 'FBI'
}


def normalize_entity(e):
    for k,v in ENTITY_REWRITE.items():
        e = e.replace(k, v)
    return e


def get_entities_from_doc(doc, return_entity_types=False):
    result = []
    for e in doc.ents:
        e_label = e.label_
        if e_label in ['DATE', 'TIME', 'CARDINAL', 'PERCENT', 'ORDINAL', 'MONEY', 'QUANTITY']:
            # Can handle separately; these entities don't work well for us, e.g., "as some 300", "second night"
            continue
        e_text = normalize_entity(e.text)
        # e_text = normalize_entity(e.lemma_)
        if e.label_ == 'PERSON':
            try:
                e_text = e_text.split()[-1:][0]
            except:
                warn("Bad entity: '{}'".format(e_text))
        result.append((e_text, e_label) if return_entity_types else e_text)
    return set(result)


def get_entities(nlp, texts):
    '''
    >>> nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"]) 
    >>> get_entities(nlp, ["This is a text from 2020. It's about John Smith.", "Donald Trump was in the New York Times."])
    [{'Smith'}, {'Trump', 'New York Times'}]
    '''
    return [get_entities_from_doc(doc) for doc in nlp.pipe(texts)]


def to_json(batch):
    return [json.loads(x) for x in batch]


def extract(batch, pass_through=False, articles_key='articles', abstract_key='abstract'):
    '''
    Extracts all articles and abstracts from the batch
    '''
    articles = []
    abstracts = []
    jbatch = to_json(batch)
    for data in jbatch:
        x, y = data[articles_key], data[abstract_key]
        if isinstance(x, list):
            x = '\n'.join(x)
        articles.append(x)
        abstracts.append(y)
    return articles, abstracts, jbatch if pass_through else None


def process_batch(nlp, extract_fct, batch_id, batch, pass_through, min_entity_precision=0.0):
    articles, abstracts, jbatch = extract_fct(batch, pass_through)
    results = [d for d in match_entities(articles, abstracts, nlp) if d['entity_precision'] >= min_entity_precision]
    if pass_through:
        for i,d in enumerate(results):
            d.update(jbatch[i])
    return results
    

def match_entities(articles, abstracts, nlp):
    ents1 = get_entities(nlp, articles)
    ents2 = get_entities(nlp, abstracts)
    results = []
    progress = tqdm.tqdm(total=len(articles), ncols=120, desc='Evaluating Entity Precision: ')
    for i, (e1, e2) in enumerate(zip(ents1, ents2)):
        unmatched = e2 - e1
        Z = len(e2) or 1e-24
        d = {
                'entities_abstract': sorted(list(e2)),
                'entities_article': sorted(list(e1)),
                'unmatched': sorted(list(unmatched)),
                'entity_precision': 1.0 - float(len(unmatched)) / Z
            }
        results.append(d)
        progress.update(1)
    progress.close()
    return results


def compute_entity_precision(data_list, output_dir):
    abstracts = [item["generated_summary"] for item in data_list]
    articles = [" ".join(item["documents"]) for item in data_list]

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
    all_results = match_entities(articles, abstracts, nlp)

    ep = [result["entity_precision"] for result in all_results]

    overall_result = {
        'mean': np.mean(ep), 
        'median': np.median(ep), 
        'variance': np.var(ep),
        'min': np.min(ep), 
        'max': np.max(ep), 
        'len': len(ep)
    }

    results = {"overall": overall_result, "separate": all_results}

    with open(os.path.join(output_dir, "entity.json"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    with open("./results.json", 'r', encoding='utf-8') as f:
        debugs = json.loads(f.read())
    result = compute_entity_precision(debugs, "entity.json")