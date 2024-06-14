import argparse
from tqdm import tqdm
import json
import time
import random
import re
import torch
from retnet.modeling_retnet import RetNetForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

random.seed(42)

keys = [str(random.randint(10**12, 10**13 - 1)) for _ in range(100)]

names = [
    "佐藤健", "鈴木一郎", "高橋恵子", "田中美咲", "渡辺直人", "伊藤博文", "山本葵", "中村優一", "小林直樹", "加藤美優",
    "吉田輝星", "山田花子", "佐々木希", "山口達也", "松本潤", "井上陽水", "木村拓哉", "林原めぐみ", "斎藤工", "前田敦子",
    "遠藤新菜", "岡田凛", "長谷川博己", "原田美枝子", "福田明日香", "森鴎外", "阿部寛", "宮崎駿", "村上春樹", "島崎遥香",
    "橋本環奈", "中谷美紀", "桜井日奈子", "小泉進次郎", "安倍晴明", "川端康成", "樋口一葉", "夏目漱石", "芥川龍之介", "有島武郎",
    "野口英世", "椎名林檎", "谷崎潤一郎", "三島由紀夫", "大江健三郎", "石原裕次郎", "北原白秋", "谷川俊太郎", "黒柳徹子", "綾瀬はるか",
    "宮沢賢治", "池波正太郎", "星新一", "小松左京", "司馬遼太郎", "尾崎豊", "浅野忠信", "坂本龍馬", "徳川家康", "織田信長",
    "豊臣秀吉", "明智光秀", "武田信玄", "上杉謙信", "毛利元就", "北条政子", "足利尊氏", "細川ガラシャ", "伊達政宗", "島津義弘",
    "真田幸村", "本多忠勝", "黒田官兵衛", "立花宗茂", "前田慶次", "直江兼続", "大谷吉継", "徳川光圻", "伊藤博文", "山県有朋",
    "渋沢栄一", "福沢諭吉", "夏目金之助", "樋口一葉", "二葉亭四迷", "国木田独歩", "夢野久作", "中島敦", "平塚らいてう", "岡本かの子",
    "与謝野晶子", "三浦綾子", "川端康成", "安藤忠雄", "伊東豊雄", "内田百閒", "横山大観", "野田秀樹", "野村克也", "室伏広治"
]

dishes = [
    "寿司", "刺身", "天ぷら", "うどん", "そば", "おにぎり", "焼き鳥", "すき焼き", "しゃぶしゃぶ", "お好み焼き",
    "たこ焼き", "とんかつ", "カレーライス", "牛丼", "ラーメン", "味噌汁", "煮物", "鍋物", "ちらし寿司", "海鮮丼",
    "餃子", "焼肉", "ひつまぶし", "お茶漬け", "かつ丼", "親子丼", "ギョーザ", "アンコウ鍋", "イカ焼き", "サンマの塩焼き",
    "サバの味噌煮", "きんぴらごぼう", "鯖寿司", "たい焼き", "かき氷", "和菓子", "どら焼き", "あんみつ", "抹茶アイスクリーム", "芋けんぴ",
    "いなり寿司", "鯛の塩焼き", "豚の角煮", "もつ煮", "黒豆", "おせち料理", "納豆", "桜餅", "わらび餅", "柏餅",
    "草もち", "抹茶", "ほうじ茶", "たい焼き", "月見うどん", "山菜おこわ", "肉じゃが", "筑前煮", "鳥の唐揚げ", "塩鮭",
    "お雑煮", "赤飯", "たらの芽", "ごま豆腐", "そうめん", "冷やし中華", "たぬきうどん", "きつねうどん", "ざるそば", "海苔巻き",
    "豚まん", "おでん", "ハトシ", "イカの塩辛", "味噌おでん", "栗きんとん", "五目寿司", "ホタルイカ", "いかすみそば", "鮭とば",
    "いわしの梅煮", "フグの唐揚げ", "ぶり大根", "イモの天ぷら", "にしんそば", "赤だし", "白和え", "ごぼう天うどん", "いか焼き", "酢の物",
    "イカの姿焼き", "イカの塩辛", "あじの開き", "カツオのたたき", "マグロの刺身", "わかめごはん", "ねぎとろ丼", "エビフライ", "ホタテバター焼き", "チキン南蛮"
]

spots = [
    "東京タワー", "京都の金閣寺", "富士山", "広島平和記念公園", "奈良公園", "東京スカイツリー", "札幌時計台", "大阪城", "姫路城", "松本城",
    "伊勢神宮", "鳥取砂丘", "宮島", "阿蘇山", "白川郷", "高千穂峡", "屋久島", "日光東照宮", "箱根", "沖縄美ら海水族館",
    "北海道旭山動物園", "富士急ハイランド", "熱海", "別府地獄めぐり", "軽井沢", "横浜中華街", "淡路島", "富良野", "那須高原", "長崎原爆資料館",
    "登別温泉", "金沢の兼六園", "八丈島", "屋久島", "琵琶湖", "種子島", "秩父", "秋葉原", "日本アルプス", "日本武道館",
    "稚内", "立山黒部アルペンルート", "国立西洋美術館", "国立科学博物館", "奄美大島", "琉球王国のグスク及び関連遺産群", "高野山", "足尾銅山", "浅草寺", "築地市場",
    "新潟の佐渡島", "仙台の青葉城", "津軽海峡", "根室半島", "名古屋城", "小笠原諸島", "鬼怒川温泉", "相模原", "八ヶ岳", "神戸港",
    "仙台の定禅寺通り", "福岡の大濠公園", "高知城", "三陸復興国立公園", "和歌山の熊野古道", "草津温泉", "道頓堀", "東京ディズニーリゾート", "USJ (ユニバーサル・スタジオ・ジャパン)", "霧島温泉",
    "横浜ランドマークタワー", "甲府の武田神社", "飛騨高山", "四国八十八箇所巡り", "二条城", "東京ドーム", "渋谷スクランブル交差点", "鎌倉大仏", "佐賀の吉野ヶ里歴史公園", "上高地",
    "清水寺", "石見銀山", "徳島の阿波踊り", "富士五湖", "下呂温泉", "九十九島", "五島列島", "小樽運河", "皇居", "奈良の東大寺",
    "神戸の異人館", "室蘭工業地帯", "サファリパーク", "名古屋のテレビ塔", "沖縄の首里城", "石垣島", "白神山地", "有馬温泉", "天橋立", "三重の伊勢神宮"
]

animes = [
    "鋼の錬金術師", "進撃の巨人", "ワンピース", "ナルト", "デスノート", "僕のヒーローアカデミア", "東京喰種", "ソードアート・オンライン", "スラムダンク", "ドラゴンボール",
    "ハンターハンター", "エヴァンゲリオン", "ジョジョの奇妙な冒険", "鬼滅の刃", "コードギアス", "攻殻機動隊", "カウボーイビバップ", "フルーツバスケット", "ブリーチ", "クレヨンしんちゃん",
    "ドラえもん", "あんさんぶるスターズ！", "ヴァイオレット・エヴァーガーデン", "RE:ゼロから始める異世界生活", "名探偵コナン", "テニスの王子様", "黒子のバスケ", "FAIRY TAIL", "ノラガミ", "ダンジョンに出会いを求めるのは間違っているだろうか",
    "銀魂", "モブサイコ100", "メイドインアビス", "ハイキュー!!", "おそ松さん", "彼方のアストラ", "新世紀エヴァンゲリオン", "とある魔術の禁書目録", "とある科学の超電磁砲", "俺ガイル",
    "物語シリーズ", "青の祓魔師", "ONE OUTS", "約束のネバーランド", "機動戦士ガンダム", "サムライチャンプルー", "無職転生", "灰と幻想のグリムガル", "マギ", "ローゼンメイデン",
    "赤髪の白雪姫", "デュラララ!!", "ペルソナ5", "デート・ア・ライブ", "サイコパス", "シュタインズ・ゲート", "リゼロ", "ニセコイ", "暗殺教室", "アルスラーン戦記",
    "ブラッククローバー", "ラブライブ!", "デジモンアドベンチャー", "ユーリ!!! on ICE", "マクロスシリーズ", "ベルセルク", "ゴールデンカムイ", "モンスター", "カードキャプターさくら", "とらドラ！",
    "オーバーロード", "カードファイト!! ヴァンガード", "ソウルイーター", "ハウルの動く城", "天空の城ラピュタ", "風の谷のナウシカ", "もののけ姫", "となりのトトロ", "千と千尋の神隠し", "天気の子",
    "君の名は。", "バッカーノ！", "キングダム", "デュラララ!!", "鉄血のオルフェンズ", "艦これ", "冴えない彼女の育てかた", "バキ", "一週間フレンズ。", "ゴールデンタイム",
    "エルフェンリート", "クラナド", "呪術廻戦", "バカとテストと召喚獣", "四月は君の嘘", "Charlotte", "エンジェルビーツ！", "ガールズ&パンツァー", "リトルバスターズ！", "ラブひな"
]


category_dict = {
    "key": keys,
    "names": names,
    "dishes": dishes,
    "spots": spots,
    "animes": animes
}

initial_prompt_templates = {
    "names": "私の名前は{word}です。",
    "dishes": "私が食べたものは{word}です。",
    "spots": "私が行った場所は{word}です。",
    "animes": "私が見たアニメは{word}です。"
}

final_prompt_dict = {
    "names": "\n私の名前は",
    "dishes": "\n私が食べたものは",
    "spots": "\n私が行った場所は",
    "animes": "\n私が見たアニメは"
}

default_prompt = "草は緑です。空は青いです。太陽は黄色いです。"

entire_prompt_template = """多くの不要なテキストの中に重要な情報が隠されています。それを見つけて覚えてください。後で重要な情報についてクイズを出します。

{initial_prompt}これを覚えてください。

{repete_prompt}
"""

garbage = "草は緑です。空は青いです。太陽は黄色いです。"
garbage = "".join([garbage] * 500)
task_description = "多くの不要なテキストの中に重要な情報が隠されています。それを見つけて覚えてください。後でこの重要な情報についてクイズを出します。\n"

def get_initial_prompt(category, word):
    initial_prompt = initial_prompt_templates[category].format(word=word)
    default_prompt_with_repeats = default_prompt * 1
    final_prompt = final_prompt_dict[category]
    entire_prompt = entire_prompt_template.format(initial_prompt=initial_prompt, repete_prompt=default_prompt_with_repeats, final_prompt_dict=final_prompt)
    return entire_prompt

def generate_prompt_landmark(model, tokenizer, category, word, max_length=64, n_suffix=0):
    task_description = "多くの不要なテキストの中に重要な情報が隠されています。それを見つけて覚えてください。後でこの重要な情報についてクイズを出します。\n"
    initial_prompt = initial_prompt_templates[category].format(word=word) 
    information_line = initial_prompt + "これを覚えてください。" + initial_prompt + "\n"
    final_question = final_prompt_dict[category]

    n_token_prefix = len(tokenizer.encode(task_description)) + \
                        len(tokenizer.encode(information_line)) + \
                        len(tokenizer.encode(final_question))
    garbage_ids = tokenizer.encode(garbage)

    available_space_for_garbage = max_length - n_token_prefix
    
    if len(garbage_ids) > available_space_for_garbage:
        garbage_ids = garbage_ids[:available_space_for_garbage - n_suffix]

    garbage_inf = tokenizer.decode(garbage_ids)

    lines = [
        task_description,
        information_line,
        garbage_inf,
        final_question,
    ]
    
    prompt = "".join(lines)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    #print(f"Prompt has {input_ids.shape[-1]} tokens")

    #print(prompt)

    answer_ids = tokenizer.encode(word, return_tensors="pt")

    generation_output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=answer_ids.shape[-1], 
        pad_token_id=tokenizer.eos_token_id,
    )

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

    is_correct = (model_answer == answer_ids[0]).all().item()
    #print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    input_text = tokenizer.decode(generation_output[0].cpu())
    model_answer = tokenizer.decode(model_answer.cpu())

    torch.cuda.empty_cache()

    return is_correct, input_text, model_answer

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Spiral-AI/RetNet-3b", choices=["Spiral-AI/RetNet-3b", "cyberagent/open-calm-3b"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--category", type=str, default="dishes")
    args = parser.parse_args()

    model_name = args.model_name
    device = args.device
    revision = args.revision
    category = args.category

    if "RetNet-3b" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
        if revision:
            model = RetNetForCausalLM.from_pretrained(model_name, revision=revision)
            model_name = "RetNet-3b" + "_" + revision.split("-")[1]
        else:
            model = RetNetForCausalLM.from_pretrained(model_name)
            model_name = "RetNet-3b"

    elif "open-calm-3b" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_name = "open-calm-3b"

    if category in category_dict:
        words = category_dict[category]
    else:
        print("Invalid category")

    model.to(device)
    model.eval()

    max_length_list = [64, 256, 512, 1024, 2048]

    dic = {"cnt": [], "input_text": [], "model_answer": [], "time": []}

    for max_length in max_length_list:
        print(f"Max Length={max_length}")

        cnt = 0
        input_texts = []
        model_answer_list = []

        start_time = time.time()

        for word in tqdm(words):
            set_seed(42)
            is_correct, input_text, model_answer = generate_prompt_landmark(model, tokenizer, category=category, word=word, max_length=max_length)
            
            input_texts.append(input_text)
            model_answer_list.append(model_answer)

            if is_correct:
                cnt += 1
        
        elapsed_time = time.time() - start_time

        print(cnt)
            
        dic["cnt"].append(cnt)
        dic["input_text"].append(input_text)
        dic["model_answer"].append(model_answer_list)
        dic["time"].append(elapsed_time)

    if revision:
        with open(f"output_{model_name}_{category}.json", "w", encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
    else:
        with open(f"output_{model_name}_{category}.json", "w", encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)