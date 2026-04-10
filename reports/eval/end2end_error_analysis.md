# End-to-End Error Analysis

- Missed detection images: `1436`
- Wrong recognition matches: `5518`
- Correct matches: `45`

## Missed Detections

| Image | Missed Count | GT Examples |
|---|---:|---|
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_003670.jpg | 3 | 螺中王, 饼, 二食堂 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011859.jpg | 7 | 天天, Tiantian, 农家土菜, 外卖电话：, 64480138 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012184.jpg | 2 | 光臨, elcome |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012576.jpg | 7 | 門, CTM, 全国连锁, 400-313-323, CTM |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008599.jpg | 2 | 全国加盟热线：, 400-9966-567 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014227.jpg | 4 | MORE, THAN, 藤椒, 钵钵鱼 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011544.jpg | 5 | 黑牛烤肉料理店, JIU, TIAN, HOWE, BBO |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000203.jpg | 2 | 欢迎光临, 福 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019639.jpg | 5 | wen, shu, de, xiao, guan |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001038.jpg | 6 | 面之缘, 面馆, Face, of, the |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 4 | 麻, TEL：, 麻辣涮串, 蜀品汇 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007221.jpg | 2 | 137898845, 电话 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006739.jpg | 4 | TOWN, 小镇咖啡, TEL:, thing |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001851.jpg | 2 | 5022234, 订餐电话： |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007998.jpg | 6 | 鲜动力, ®, X·POWER, 品牌热线：, 4008-330-627 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_015080.jpg | 1 | ® |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007915.jpg | 1 | 小智 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014611.jpg | 3 | 由理发店上二楼, 旋转小火锅, 轩月阁 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_015949.jpg | 2 | 涮烤主, 乌鱼 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011916.jpg | 3 | 肠粉, 卤粉, 热卤 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014418.jpg | 4 | 峰鸭头, FENGYATOU, 加盟电话, 15968869009 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019548.jpg | 6 | ®, 步大叔, Uncle, 老式麻辣烫, 加盟电话 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000473.jpg | 5 | 牛扒, 自助餐, 简餐, 咖啡, 饮品 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005876.jpg | 2 | 奶鋪, 甜 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012766.jpg | 9 | 甜品站, TOFU, PUDDING, DIARY, DESSERT |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006222.jpg | 5 | 家常菜馆, 快餐盒饭炝拌肉肉汤饭水饺刀削面油泼面冷面凉拌面, 狗, 司, 狗 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010016.jpg | 1 | 八号椒 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_002970.jpg | 1 | 加盟 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007179.jpg | 4 | 恒, 兼营, 子果饼煎子嘴卫, 津门 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_009322.jpg | 2 | 约, 定 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013145.jpg | 1 | 小侑宴 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019290.jpg | 2 | 1857817, YEFU |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017493.jpg | 2 | 休, 啃吧 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014930.jpg | 1 | 餐馆 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001700.jpg | 6 | DAIJINJI, Snurce, to, 1758, 米饭 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_016772.jpg | 2 | 东, 奇 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011407.jpg | 1 | 皇上请选锅 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_002733.jpg | 5 | 新, 淼鑫, ®, 四季养胃营养配方, 最高可享78 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019584.jpg | 4 | 粥, 欢, 35638, 外卖 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017462.jpg | 3 | 真, 清真, 早餐供应 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006910.jpg | 3 | 岗上渣渣, ®, 重庆人的味觉乡愁 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013106.jpg | 1 | TEL |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007836.jpg | 12 | 成都, 花房, 串串, 音乐餐吧, 本座三层 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013938.jpg | 3 | 我, AN, THE |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017353.jpg | 4 | 饮料, 老店, ®, 老字號 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_004454.jpg | 2 | 面皮, 麻辣烫 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007480.jpg | 18 | 聘, 小, 饼价目表, 鲜肉饼, 2元 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006558.jpg | 2 | 拉条子, 扯面 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_002997.jpg | 1 | 旭 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_016250.jpg | 2 | Tea, Milk |

## Wrong Recognitions

| Image | GT | Pred | IoU |
|---|---|---|---:|
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_003670.jpg | 一餐饮 | 砂家· | 0.9487 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_003670.jpg | 中一餐饮 | 功一中 | 0.8609 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_003670.jpg | 螺蛳粉 | 古面香 | 0.8519 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_009258.jpg | 厦门特香食品 | 古子串冒飲 | 0.9267 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011859.jpg | 紫街店 | 砂粉 | 0.9524 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011859.jpg | 天天煲仔 | 外子 | 0.9337 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011859.jpg | 玲 | 味 | 0.8350 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_009448.jpg | 便当 | 黑都 | 0.8263 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_009448.jpg | 优食光 | 粥 | 0.6336 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012184.jpg | 馋火炉魚 | 豆七馆 | 0.6905 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012184.jpg | 鑪 | 家 | 0.6002 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012576.jpg | 朝天門 | 串串 | 0.8085 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012576.jpg | 中国·重庆·老字号 | A | 0.7891 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012576.jpg | 火锅·沈阳·奥体店 | 饭 | 0.7885 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012576.jpg | 朝天 | 八多 | 0.5035 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008599.jpg | 烤肉拌饭 | 张小排 | 0.9583 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008599.jpg | 张秀梅 | 蜀花 | 0.9388 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008599.jpg | 脆皮鸡饭 | 甜豆部 | 0.9336 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008599.jpg | 黄 | 菜 | 0.8828 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011108.jpg | 谢记烧烤 | 粉生饼 | 0.9395 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014227.jpg | 猫赞 | 麻辣饼 | 0.5877 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_011544.jpg | 九田家 | 辣动 | 0.9576 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000203.jpg | 九大簋家宴 | 营重中子 | 0.8715 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000203.jpg | 乙末年金秋 | 老 | 0.5565 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005149.jpg | 中大店 | 刘中 | 0.9906 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005149.jpg | 串串锅 | 鲜多烤 | 0.9130 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005149.jpg | 那一年我们经过乐山 | r！ | 0.8850 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005149.jpg | 乐吃 | 回！ | 0.7390 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 绝味笋子面 | 面常削品 | 0.8988 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 张记 | 米记 | 0.8608 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 炒饭 | 福风 | 0.8536 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 盖饭、 | 小香 | 0.7330 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 特色小炒、 | 小豆牛 | 0.7257 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010732.jpg | 绝味笋 | 麻辣鱼 | 0.6633 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019639.jpg | 文叔的小館 | 4吃 | 0.7729 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019639.jpg | 文叔的小館 | 秘牌味吉 | 0.5857 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001038.jpg | 只吃一次是我的错 | 店 | 0.9489 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001038.jpg | 面之 | 天岛 | 0.9268 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001038.jpg | 一次不吃是你的错 | 黑的香— | 0.8965 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001038.jpg | 特色面 | 正菜 | 0.5392 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 老成都味道 | 承物淇时茶 | 0.9227 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 香菜牛肉 | 时线蛋 | 0.8971 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 麻辣英 | 烧粉 | 0.8935 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 特色小郡肝 | 中脆（号头 | 0.8868 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 麻辣牛肉 | 肉牛辣线 | 0.8752 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000754.jpg | 86666066 | 美道 | 0.5198 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007221.jpg | 鲜之道 | 麻之锅 | 0.9126 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007221.jpg | 锡纸 | 麻锅 | 0.9083 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_007221.jpg | 花甲粉 | 辣王 | 0.7320 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006739.jpg | SMALL | TE | 0.9306 |

## Correct Samples

| Image | Text | IoU |
|---|---|---:|
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_002794.jpg | 面 | 0.6254 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013964.jpg | ® | 0.8642 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_001230.jpg | ® | 0.6099 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_018149.jpg | ® | 0.9048 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_003367.jpg | ® | 0.8769 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000786.jpg | 面 | 0.9010 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_002193.jpg | ® | 0.7757 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000077.jpg | 排 | 0.7408 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008681.jpg | ® | 0.7670 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_000880.jpg | ® | 0.7727 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019933.jpg | ® | 0.8288 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012337.jpg | ® | 0.7000 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013419.jpg | 中 | 0.7972 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005694.jpg | ® | 0.7888 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017709.jpg | 906 | 0.7312 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_004953.jpg | 餐 | 0.5289 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_012875.jpg | ® | 0.8444 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017858.jpg | ® | 0.8868 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_019300.jpg | ® | 0.7171 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_009793.jpg | 品 | 0.9668 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_010742.jpg | ® | 0.6260 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_005165.jpg | 中 | 0.8873 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_018547.jpg | ® | 0.7870 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_006643.jpg | ® | 0.9011 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014836.jpg | 真 | 0.8165 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_016372.jpg | ® | 0.8159 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_014238.jpg | 海鲜 | 0.9071 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_017561.jpg | ® | 0.7586 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_008592.jpg | 品 | 0.8285 |
| /root/autodl-tmp/dl-text-recoginze/data/raw/rects/train/img/train_ReCTS_013791.jpg | ® | 0.7812 |
