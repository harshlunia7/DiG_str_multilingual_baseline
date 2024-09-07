from torch.utils.data import Dataset
from torchvision import transforms
import torch

import lmdb
import string
import six
import numpy as np
from PIL import Image, ImageFile
import cv2

from transforms import CVColorJitter, CVDeterioration, CVGeometry

from imgaug import augmenters as iaa
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0) # cv2's multiprocess will impact the dataloader's workers.

class ImageLmdb(Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform,
               use_aug=False, use_abi_aug=False, use_color_aug=False):
    super(ImageLmdb, self).__init__()

    self.env = lmdb.open(root, max_readers=32, readonly=True)
    self.txn = self.env.begin()
    self.nSamples = int(self.txn.get(b"num-samples"))

    num_samples = num_samples if num_samples > 1 else int(self.nSamples * num_samples)
    self.nSamples = int(min(self.nSamples, num_samples))

    self.root = root
    self.max_len = max_len
    self.transform = transform
    self.use_aug = use_aug
    self.use_abi_aug = use_abi_aug
    self.use_color_aug = use_color_aug
    if use_aug:
      if use_abi_aug:
        mean = std = 0.5
        self.augment_abi = transforms.Compose([
              CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
              CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
              CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
              transforms.Resize((32, 128), interpolation=3),
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=torch.tensor(mean),
                  std=torch.tensor(std))
          ])
      else:
        # augmentation following seqCLR
        if use_color_aug:
          self.augmentor = self.color_aug()
        else:
          self.augmentor = self.sequential_aug()
        mean = std = 0.5
        self.aug_transformer = transforms.Compose([
              transforms.Resize((32, 128), interpolation=3),
              transforms.RandomApply([
                  transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
              ], p=0.8),
              transforms.RandomGrayscale(p=0.2),
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=torch.tensor(mean),
                  std=torch.tensor(std))
          ])

    # Generate vocabulary
    types_of_voc = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'CHINESE']
    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'CHINESE']
    self.classes = self._find_classes(voc_type)
    self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
    self.idx_to_class = dict(zip(range(len(self.classes)), self.classes))
    self.use_lowercase = (voc_type == 'LOWERCASE')
    self.normalize = (voc_type in types_of_voc[:-1])

  def _find_classes(self, voc_type, EOS='EOS',
                    PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'CHINESE']
    if voc_type == 'LOWERCASE':
      # voc = list(string.digits + string.ascii_lowercase)
      voc = list('0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    elif voc_type == 'ALLCASES':
      voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
      voc = list(string.printable[:-6])
    elif voc_type == 'CHINESE':
      voc = list(' !"#%&\'()+,-./0123456789:;=>@ABCDEFGHIJKLMNOPQRSTUVWXYZ\\]_`abcdefghijklmnopqrstuvwxyz|~®°·Λ—‘’“”•…←→Ⓡ▪●★、。〇《》【】の一丁七万三上下不与丑专且世丘丙业东丝两严丨个丫中丰串临丶丸丹为主丽举久么义之乌乐乒乓乔乘乙九也习乡书买乱乳乾了予争事二于亏云互五井亚交亦产亨享京亭亮亲人亻亿什仁仅今介从仑仓仔仕他付仙仟代令以仪们仰仲件价任份仿企伊伍伏休众优伙会伞伟传伤伦伪伯估伴伸似伽位低住佐佑体何余佛作你佩佬佰佳使來侈例供依侠侣侧侨侬侯便促俄俊俏保信俪俬修俱倍倒候借倡倩倪值倾假偉做停健偿傅储催傲像優儿允元兄充兆先光克免兑兔党入全八公六兰共关兴兵其具典兹养兼兽兿冀内冈冉册再冒写军农冠冬冯冰冲决况冶冷冻净准凉凌减凝几凡凤凭凯凰出击函刀刁分切刊刑划列刘则刚创初刨利别刮到制刷券刹刺刻剂削前剑剔剧剪副割創力劝办功加务动助励劲劳劵势勇勒勘務募勤勺勾勿包化北匙匠匹区医匾十千升午卉半华协卑卒卓单卖南博卜占卡卢卤卧卫印危即卷卸卿厂厅历压厕厘厚原厢厦厨去县叁参又叉及友双反发叔取受变叠口古另只叫召叭叮可台史右叶号司吃各合吉吊同名后向吕吗君吞否吧含听启吴吸吹吾呀呆呈告员周味呵呷呼命和咏咕咖咚咨咪咸咽品哈响哥哦哨哪哲哺唇唐唛售唯唱商啊啡啤啥啦善喆喇喉喜喝喵喷喻嗨嘉嘎嘟嘴器囍四回因团园困围固国图圃圆圈國園圖團土圣在圭地圳场圾址均坊坏坐坑块坚坛坝坠坡坤坦坪垂垃型垫埃城埔域埠培基堂堆堡堤堰場堵塑塔塘塞填境墅墓墙增墨墩壁士壮声壳壶壹处备复夏夕外多夜够大天太夫央失头夷夹奇奈奉奋奔奕奖套奠奢奥女奶她好如妃妆妇妈妍妙妞妥妮妹妻姆始姐姓委姚姜姨姬姻姿威娃娅娇娘娜娟娱婆婚婴婷媒媛嫁嫂嫩子孔孕字存孙孚孝孟季孤学孩孵學宁它宅宇守安宋完宏宗官宙定宜宝实宠审客宣室宫宰害宴宵家宸容宽宾宿寄密寇富寒寓察寨寳寶寸对寺寻导寿封射将尊小少尔尖尘尚尝尤尧就尹尺尼尽尾尿局层居屈届屋屏展属屯山屹岁岐岗岙岛岩岭岳岸峡峨峰崂崇崧嵌川州巡巢工左巧巨巩巫差己已巴巷巾币市布帅帆师希帐帕帘帛帜帝带席帮常帽幅幕幢干平年并幸幻幼广庄庆床序库应底店庙府废度座庭庵康廉廊廖廣延廷建开异弄式弓引弗弘弟张弥弧弯弱張弹强归当录形彤彦彩彪彬彭影役彻往征径待很律徐徒得御微德徽心必忆志忘忙忠忧快念怀态怕思怡急性怪总恂恋恒恢恩恭息恶悟悠患悦您情惊惑惠想愉意愛感愿慈慎慕慢慧慶懂戈戏成我戒或战戴户房所扁扇手扌才扎扑扒打托扣执扩扫扬扭扯扶批找承技抄把抓投抗折抚抛抢护报披抱抵押抽担拆拇拉拌拍拐拒拓拔拖招拜拟拥拨择拳拼拾拿持挂指按挑挖挡挥振捆捐捞损换据捷授掉掌排掘探接控推掺描提插揭援揽搏搬搭携摄摆摇摊摩撑撕撞播撸擀操擦攀支收改放政故效敏救教敢散敦敬数整文斋斌斑斗料斜斤断斯新方施旁旅旋族旗无日旦旧旨早旭旱时旺昂昆昇昊昌明易昕星映春昭是昶显時晋晒晓晕晖晚晟晨普景晴晶智晾暑暖暨暴曙曲更書曹曼曾最會月有朋服朗望朝期木未末本术朱朴朵机杀杂权杆杉李杏材村杜杞束杠条来杨杭杯杰東杷松板极构枕林果枝枣枪枫架枸柏柒染柔柚柜柠查柯柱柳柴柿栅标栈栋栏树栓栖栗校样核根格栽桂桃框案桌桐桑桔档桥桦桩桶梁梅梦梧梨梭梯械梵检棉棋棍棒棕棚棠森棵棺椅植椎椒椰椿楊楚楠業楼概榆榔榕榜榨榴榻槎槐槟槽樂樊樓樟模横樱橙橡橱檀檬欠次欢欣欧款歌歡止正此步武歲殊残殖殡段殿毂毅母每毒比毛毫毯氏民气氟氣氧氨氩水氵永汀汁求汇汉汕汗汛江池污汤汪汽汾沁沂沃沈沉沌沐沙沟没沥沧沪沫河油治沿泉泊泓法泗泛泡波泥注泰泳泵泸泼泽泾洁洋洒洗洛洞津洪洱洲活洼洽派流浆浇测济浏浓浙浜浦浩浪浮浴海涂消涌涛涤润涮涯液涵淀淇淋淑淘淞淡淮深淳混添淼清渔渝渡渣温港游湖湘湛湯湾湿源溢溪滇滋滑滘滙滚满滤滨滩滴漂漆漏演漕漢漫漳潍潘潜潢潭潮潼澄澜澡澳激濠瀘瀚灌火灭灯灰灵灶灸灾灿炉炎炒炖炝炫炭炮炸点烂烈烘烙烛烟烤烧烨烩烫热烹烽焊焕焖焗焙無焦然煌煎煤照煨煮煲熊熏熙熟熨熹燃燎燕爆爪爬爱爵爸爽片版牌牙牛牡牢牧物牵特犬犯状狂狐狗独狮狸狼猪猫献猴玄率玉王玖玛玩玫环现玲玺玻珀珊珍珑珞珠班球理琥琦琪琳琴琶瑜瑞瑰瑶璃璜瓜瓦瓶瓷甘甜生用甩甫田由甲申电男甸画畅界留畜略番畫當疆疏疑疗疣疤疫疮疯疹疼疾病症痒痔痕痘痛痣痧瘊瘤瘦癣登發白百的皆皇皓皖皮盆盈益盐监盒盔盖盗盘盛盟目盱盲直相盼盾省眉看眙真眠眼着睛睡督睦睫睿瞳知矫短石矾矿码砂研砖砚破础硅硒硕硚硫硬确碍碎碑碗碟碧碰碱碳磁磊磨磷示礼社祖祛祝神祠祥票祺禁禄禅福禧禹离禽禾秀私秋种科秒秘租秤秦积称移稀程税稚種稳稻穆穗穴究空穿突窖窗窝立站竞章童端竹竿笋笑笔笛笨第笼等筋筑筒答策筛筝筷筹签简箕算管箭箱節篮篷簧籍米类籽粉粑粒粗粘粤粥粮粱粽精糊糕糖糯系紅素索紧紫絮緣繁纟纠红纤约级纪纫纯纱纳纶纷纸纹纺纽线练组绅细织终绍经绒结绕绘给络绝统绣绥继绩续绳维绵绸综绽绿缆缇缔编缘缝缤缩缴缸缺罐网罗罚罡罩罪置署羅羊美羔群義羽翅翔翟翠翡翰翻翼耀老考者而耐耕耗耳聊职联聘聚聪肃肇肉肌肖肘肚肝肠股肤肥肩肯育肴肺肾肿胀胃背胎胖胜胞胡胶胸能脂脆脉脊脑脖脚脱脸腊腐腔腩腰腱腹腺腻腾腿膏膜膳臊臣臨自臭至致臻興舌舍舒舞舟航舰舶船艇艮良色艳艺艾节芋芒芙芜芝芦芬芭芯花芳芹芽苍苏苑苗若苦英苹茂范茄茅茗茜茨茵茶茸荆草荐荒荔荘荞荟荣荤药荷莉莊莎莓莞莫莱莲获莹菁菇菊菌菜華菱菲萄萌萍萝营萧萨萬落葉著葡董葫葬葱葵蒂蒋蒙蒜蒲蒸蓄蓉蓓蓝蓬蔓蔚蔡蔬蕴蕾薄薇薛薪薯藍藏藕藝藤藻蘭虎虚號虫虹虾蚁蚂蚊蚕蚝蛋蛙蛛蛳蜀蜂蜓蜘蜜蜡蜻蝎蝶融螺蟠蟹血行術衔街衡衣补表衫衬衰袁袋袍袖袜被袭裁装裕裙裤裱裳褔褥襄西要覆见观规视览觉角解触言記設誉誠警计订认让训议讯记讲许论讼设访证评识诉诊译试诗诚话诞询详语诱说请诸诺读课调谅谈谊谐谢谦谨谱谷豆豐象豪豫豹貂財貢貴賀贝负贡财责贤账货质贩贫购贯贴贵贷贸费贺贾赁资赌赏赐赔赖赚赛赞赠赢赣赤赫走赵起超越趟趣足跃跆跌跑距跟跨路跳践踏踩蹄蹈蹭身車軒车轨轩转轮软轴轻载轿辅辆辉辑输辛辣辫辰边辽达迁迅过迈迎运近返还这进远违连迪迮迷迹追退送适逆选逍透递途逗通速造逢連進逸遇運道達遗遣遥遮遵避邀邑邓邢那邦邮邯邱邵邸邹邻郁郊郎郏郑郝郡部郭郸都鄂酉配酒酥酬酱酵酷酸酿醇醉醋醛采釉释里重野量金釜鉴銀錦鑫针钉钓钛钜钟钢钣钥钦钰钱钵钻铁铂铃铜铝铣铭铲铵银铸铺链销锁锂锅锈锋锌锐错锡锣锦键锯锻镀镁镇镐镜镶長长門開間閣门闪闭问闯闲间闵闷闸闺闻闽阀阁阅阜队防阳阴阶阻阿陀陂附际陆陇陈降限陕院除险陪陵陶陷陽隆随隐隔障隧难雀雁雄雅集雍雕雨雪雯雲零雷電雾需震霖霞露霸青靓靖静非靠面革鞋鞍鞭韦韩韭音韵韶順頭页顶项顺须顾顿颁颂预领颈颐频颖颗题颜额風风飘飞食飯餐館饨饭饮饰饱饲饵饶饸饹饺饼饿馄馅馆馈馋馍馒首香馨馬駕马驭驰驴驶驹驻驼驾驿骆验骏骑骗骨高髙髪鬼魂魅魏魔魚鱼鱿鲁鲍鲜鲤鲫鳳鴻鷄鸟鸡鸣鸭鸽鸿鹅鹉鹏鹤鹦鹰鹿麗麟麦麵麺麻黄黎黑黔默黛鼎鼓鼠鼻齐齿龄龍龙！＃＆（）＋，－０１８：；？Ｅ｜￥')
    else:
      raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc

  def __len__(self):
    return self.nSamples

  def sequential_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Crop(percent=((0, 0.3),
                              (0, 0.0),
                              (0, 0.3),
                              (0, 0.0)),
                              keep_size=True),
            iaa.Crop(percent=((0, 0.0),
                              (0, 0.1),
                              (0, 0.0),
                              (0, 0.1)),
                              keep_size=True),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
            # iaa.AdditiveGaussianNoise(scale=(0, 0.15*255), per_channel=True),
            iaa.Rotate((-10, 10)),
            # iaa.Cutout(nb_iterations=1, size=(0.15, 0.25), squared=True),
            iaa.PiecewiseAffine(scale=(0.03, 0.04), mode='edge'),
            iaa.PerspectiveTransform(scale=(0.05, 0.1)),
            iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
            iaa.Grayscale(alpha=(0.0, 1.0)),
          ],
          random_order=True)
        ]
      ).augment_image,
    ])
    return aug_transform
  
  def color_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
            iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
            iaa.Grayscale(alpha=(0.0, 1.0)),
          ],
          random_order=True)
        ]
      ).augment_image,
    ])
    return aug_transform

  def open_lmdb(self):
    self.env = lmdb.open(self.root, readonly=True, create=False)
    # self.txn = self.env.begin(buffers=True)
    self.txn = self.env.begin()

  def __getitem__(self, index):
    if not hasattr(self, 'txn'):
      self.open_lmdb()

    # Load image
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]
    
    # Load label
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.use_lowercase:
      word = word.lower()

    if len(word) + 1 >= self.max_len:
      # print('%s is too long.' % word)
      return self[index + 1]
    ## fill with the padding token
    label = np.full((self.max_len,), self.class_to_idx['PADDING'], dtype=np.int)
    label_list = []
    for char in word:
      if char in self.class_to_idx:
        label_list.append(self.class_to_idx[char])
      else:
        label_list.append(self.class_to_idx['UNKNOWN'])
    ## add a stop token
    label_list = label_list + [self.class_to_idx['EOS']]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)
    if len(label) <= 0:
      return self[index + 1]
    
    # Label length
    label_len = len(label_list)

    # augmentation
    if self.use_aug:
      if self.use_abi_aug:
        aug_img = self.augment_abi(img)
      else:
        # augmentation
        aug_img = self.augmentor(np.asarray(img))
        aug_img = Image.fromarray(np.uint8(aug_img))
        aug_img = self.aug_transformer(aug_img)
      return aug_img, label, label_len
    else:
      assert self.transform is not None
      img = self.transform(img)
      return img, label, label_len
