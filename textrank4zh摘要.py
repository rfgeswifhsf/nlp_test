from textrank4zh import TextRank4Keyword,TextRank4Sentence

text = """白宫不再打算在疫情期间遣返数万名中国和其他外国学生后，又提出更极端建议：对9000万中共党员及其家属实施签证禁令。这个试探性气球很可能也被扔进垃圾桶。因为若禁令落实，（美中）进一步的贸易谈判将取消，外交对话、学术交流和科学合作也无从谈起。这还将导致往来太平洋两岸的航班商务舱空空如也，通常坐满游客的后排也会空出一大片。

这个想法行不通，却引发一些令人不安的问题。其中最主要的是：为何美国没有更严肃的关于中国的公开讨论？对国会政客而言，像签证禁令这样史无前例的提议，是在一个潜在危险真空中产生的。《金融时报》专栏作家贾南·加内什说，“身处华盛顿，会感觉到一个国家正滑向针对中国的无尽冲突，诡异的是居然鲜有（相关）讨论。”但若美国真的要与这样强大的对手摊牌，选民应该知道，如此冲突将造成巨大的国家代价和高昂的个人牺牲。且粗暴威胁会刺激中国民众的反美情绪。许多中国人都相信美国江河日下，连对抗新冠病毒的意志都缺乏，遑论通过加强经济基础和科学研究来应对一个崛起的超级大国的挑战。

伦敦大学学院创新与公共项目研究所研究员劳里·麦克法兰认为，华盛顿对华为的担忧是真实的。但在关于国家安全的言辞下，隐藏着更深层次担忧——中国经济模式有可能匹敌并威胁到长期以来支撑美国霸权的科技优势。"""

tr4w = TextRank4Keyword()

tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

print('关键词：')
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)


print()
print('关键短语：')
for phrase in tr4w.get_keyphrases(keywords_num=2, min_occur_num=2):
    print(phrase)


tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source='all_filters')

print()
print('摘要：')
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)  #

