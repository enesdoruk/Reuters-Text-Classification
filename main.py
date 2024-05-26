from dataloader import get_data
from preprocess import (removeUrlsHtmls, removeSpecialChar, removeRomanNumb, \
                        convert2lower, removeStopWords, removePunctuation, \
                        removeDigit, removewhiteSpace, remove_rare_words, \
                        tokenization, stemming, lemmatizer, remove_NanValues, \
                        split_train_test, list2dict_class)

from model import (generate_language_models, classify_documents)
from metrics import (calculate_metrics_for_all_classes, calculate_macro_average)
from vis import plot_histogram


def main(path, verbose=True):
    documents, categories = get_data(path)
    
    documents = list(map(removeUrlsHtmls, documents))
    documents = list(map(removeSpecialChar, documents))
    documents = list(map(removeRomanNumb, documents))
    documents = list(map(convert2lower, documents))
    documents = list(map(removeStopWords, documents))
    documents = list(map(removePunctuation, documents))
    documents = list(map(removeDigit, documents))
    documents = list(map(removewhiteSpace, documents))
    documents = list(map(remove_rare_words, documents))
    
    token_documents = list(map(tokenization, documents))
    token_documents = list(map(lemmatizer, token_documents))
    token_documents, categories = remove_NanValues(token_documents, categories)
    
    train_doc, train_cat, test_doc, test_cat = split_train_test(token_documents, categories)
    train_cat_dict = list2dict_class(train_doc, train_cat)
    test_cat_dict = list2dict_class(test_doc, test_cat)
    gt_label = [pred[0] for pred in test_cat]
       
    unigram_models, bigram_models, trigram_models = generate_language_models(train_cat_dict)

    classifications_no_smoothing_unig = classify_documents(test_doc, unigram_models)
    metrics_no_smt_unig = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_no_smoothing_unig], \
                                                            set(gt_label + [pred[-1] for pred in classifications_no_smoothing_unig]))
    macro_recall_no_smt_unig, macro_precision_no_smt_unig, macro_f_score_no_smt_unig = calculate_macro_average(metrics_no_smt_unig)
    
    
    classifications_with_smoothing_unig = classify_documents(test_doc, unigram_models, use_smoothing=True)
    metrics_smt_unig = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_with_smoothing_unig], \
                                                            set(gt_label + [pred[-1] for pred in classifications_with_smoothing_unig]))
    macro_recall_smt_unig, macro_precision_smt_unig, macro_f_score_smt_unig = calculate_macro_average(metrics_smt_unig)
    
    
    classifications_no_smoothing_big = classify_documents(test_doc, bigram_models)
    metrics_no_smt_big = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_no_smoothing_big], \
                                                            set(gt_label + [pred[-1] for pred in classifications_no_smoothing_big]))
    macro_recall_no_smt_big, macro_precision_no_smt_big, macro_f_score_no_smt_big = calculate_macro_average(metrics_no_smt_big)
    
    
    classifications_with_smoothing_big = classify_documents(test_doc, bigram_models, use_smoothing=True)
    metrics_smt_big = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_with_smoothing_big], \
                                                            set(gt_label + [pred[-1] for pred in classifications_with_smoothing_big]))
    macro_recall_smt_big, macro_precision_smt_big, macro_f_score_smt_big = calculate_macro_average(metrics_smt_big)
    

    classifications_no_smoothing_trig = classify_documents(test_doc, trigram_models)
    metrics_no_smt_trig = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_no_smoothing_trig], \
                                                            set(gt_label + [pred[-1] for pred in classifications_no_smoothing_trig]))
    macro_recall_no_smt_trig, macro_precision_no_smt_trig, macro_f_score_no_smt_trig = calculate_macro_average(metrics_no_smt_trig)
    
    
    classifications_with_smoothing_trig = classify_documents(test_doc, trigram_models, use_smoothing=True)
    metrics_smt_trig = calculate_metrics_for_all_classes(gt_label, [pred[-1] for pred in classifications_with_smoothing_trig], \
                                                            set(gt_label + [pred[-1] for pred in classifications_with_smoothing_trig]))
    macro_recall_smt_trig, macro_precision_smt_trig, macro_f_score_smt_trig = calculate_macro_average(metrics_smt_trig)
        
    
    if verbose:
        plot_histogram(train_cat_dict, "train")
        plot_histogram(test_cat_dict, "test")
        
        print('='*20)
        print(f'UNIGRAM - Without Smoothing; recal: {macro_recall_no_smt_unig}, precision: {macro_precision_no_smt_unig}, f1-score: {macro_f_score_no_smt_unig}')
        print(f'UNIGRAM - With Smoothing; recal: {macro_recall_smt_unig}, precision: {macro_precision_smt_unig}, f1-score: {macro_f_score_smt_unig}')
        print(f'BIGRAM - Without Smoothing; recal: {macro_recall_no_smt_big}, precision: {macro_precision_no_smt_big}, f1-score: {macro_f_score_no_smt_big}')
        print(f'BIGRAM - With Smoothing; recal: {macro_recall_smt_big}, precision: {macro_precision_smt_big}, f1-score: {macro_f_score_smt_big}')
        print(f'TRIGRAM - Without Smoothing; recal: {macro_recall_no_smt_trig}, precision: {macro_precision_no_smt_trig}, f1-score: {macro_f_score_no_smt_trig}')
        print(f'TRIGRAM - With Smoothing; recal: {macro_recall_smt_trig}, precision: {macro_precision_smt_trig}, f1-score: {macro_f_score_smt_trig}')
        print('='*20)
        
    
if __name__ == "__main__":
    path = 'reuters21578'
    main(path, verbose=True)