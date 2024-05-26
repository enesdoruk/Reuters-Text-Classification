import os 
from bs4 import BeautifulSoup


def get_data(path):
    files = [file for file in sorted(os.listdir(path)) if file.endswith('.sgm')]

    text_categories = {}
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore')
        dataFile = f.read()
        
        soup = BeautifulSoup(dataFile, 'html.parser')
        contents = soup.findAll('body')
        categories = soup.findAll('topics')
        
        for text_tag, topic_tag in zip(contents, categories):
            text = text_tag.get_text()
            categories = [category.text.strip() for category in topic_tag.find_all('d')]
            text_categories[text] = categories
            
            
    non_cat = [key for key, value in text_categories.items() if len(value) < 1]  
    for key in non_cat:
        del text_categories[key]
        

    categories_list = ["earn", "acq", "grain", "crude", "money-fx"]
    wrng_cat = []
    for txt, cls in text_categories.items():
        if len(cls) == 1 and cls[0] not in categories_list: 
            wrng_cat.append(txt)
        elif len(cls) > 1: 
            count=0
            for c in cls:
                if c in categories_list:
                    count+=1
            if count == 0 or count > 1: 
                wrng_cat.append(txt)

    for key in wrng_cat:
        del text_categories[key]        
        

    documents = list(text_categories.keys())
    categories = list(text_categories.values())

    for i, cat in enumerate(categories): 
        if len(cat) > 1:
            for cat_vl in categories_list:
                if cat_vl in cat:
                    categories[i] = [cat_vl]     
                    
    return documents, categories