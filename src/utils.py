

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_mapping():
    arxiv_mapping = {
    'arxiv cs ai': 'cs.AI',
    'arxiv cs cl': 'cs.CL',
    'arxiv cs cc': 'cs.CC',
    'arxiv cs ce': 'cs.CE',
    'arxiv cs cg': 'cs.CG',
    'arxiv cs gt': 'cs.GT',
    'arxiv cs cv': 'cs.CV',
    'arxiv cs cy': 'cs.CY',
    'arxiv cs cr': 'cs.CR',
    'arxiv cs ds': 'cs.DS',
    'arxiv cs db': 'cs.DB',
    'arxiv cs dl': 'cs.DL',
    'arxiv cs dm': 'cs.DM',
    'arxiv cs dc': 'cs.DC',
    'arxiv cs et': 'cs.ET',
    'arxiv cs fl': 'cs.FL',
    'arxiv cs gl': 'cs.GL',
    'arxiv cs gr': 'cs.GR',
    'arxiv cs ar': 'cs.AR',
    'arxiv cs hc': 'cs.HC',
    'arxiv cs ir': 'cs.IR',
    'arxiv cs it': 'cs.IT',
    'arxiv cs lo': 'cs.LO',
    'arxiv cs lg': 'cs.LG',
    'arxiv cs ms': 'cs.MS',
    'arxiv cs ma': 'cs.MA',
    'arxiv cs mm': 'cs.MM',
    'arxiv cs ni': 'cs.NI',
    'arxiv cs ne': 'cs.NE',
    'arxiv cs na': 'cs.NA',
    'arxiv cs os': 'cs.OS',
    'arxiv cs oh': 'cs.OH',
    'arxiv cs pf': 'cs.PF',
    'arxiv cs pl': 'cs.PL',
    'arxiv cs ro': 'cs.RO',
    'arxiv cs si': 'cs.SI',
    'arxiv cs se': 'cs.SE',
    'arxiv cs sd': 'cs.SD',
    'arxiv cs sc': 'cs.SC',
    'arxiv cs sy': 'cs.SY'
    }

    # arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule_Learning",
        'Neural_Networks': "Neural_Networks",
        'Case_Based': "Case_Based",
        'Genetic_Algorithms': "Genetic_Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement_Learning",
        'Probabilistic_Methods': "Probabilistic_Methods"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

    wikics_mapping = {
        'Computational linguistics': 'Computational linguistics',
        'Databases': 'Databases',
        'Operating systems': 'Operating systems',
        'Computer architecture': 'Computer architecture',
        'Computer security': 'Computer security',
        'Internet protocols': 'Internet protocols',
        'Computer file systems': 'Computer file systems',
        'Distributed computing architecture': 'Distributed computing architecture',
        'Web technology': 'Web technology',
        'Programming language topics': 'Programming language topics'
    }

    tolokers_mapping = {
        'not banned': 'not banned',
        'banned': 'banned'
    }

    twenty_newsgroup_mapping = {'alt.atheism': 'News about atheism.', 'comp.graphics': 'News about computer graphics.', 'comp.os.ms-windows.misc': 'News about Microsoft Windows.', 'comp.sys.ibm.pc.hardware': 'News about IBM PC hardware.', 'comp.sys.mac.hardware': 'News about Mac hardware.', 'comp.windows.x': 'News about the X Window System.', 'misc.forsale': 'Items for sale.', 'rec.autos': 'News about automobiles.', 'rec.motorcycles': 'News about motorcycles.', 'rec.sport.baseball': 'News about baseball.', 'rec.sport.hockey': 'News about hockey.', 'sci.crypt': 'News about cryptography.', 'sci.electronics': 'News about electronics.', 'sci.med': 'News about medicine.', 'sci.space': 'News about space and astronomy.', 'soc.religion.christian': 'News about Christianity.', 'talk.politics.guns': 'News about gun politics.', 'talk.politics.mideast': 'News about Middle East politics.', 'talk.politics.misc': 'News about miscellaneous political topics.', 'talk.religion.misc': 'News about miscellaneous religious topics.'}



    return {
        'arxiv': arxiv_mapping, 
        'citeseer': citeseer_mapping, 
        'pubmed': pubmed_mapping, 
        'cora': cora_mapping, 
        'products': products_mapping,
        'wikics': wikics_mapping,
        'tolokers': tolokers_mapping,
        '20newsgroup': twenty_newsgroup_mapping
    }