import argparse
import torch
from torchvision import datasets, transforms
from d2l import torch as d2l
import clip
from torchmetrics.image.kid import KernelInceptionDistance
import pandas as pd
import copy
import matplotlib.pyplot as plt  
import os
import PIL.Image as Image

def get_args():
    parser = argparse.ArgumentParser(description='Caculate KID score with CLIP ViT-B/32 extracting features')
    parser.add_argument('--real_path', default='./data/real/')
    parser.add_argument('--fake_path', default='./data/test/')
    parser.add_argument('--real_A_path', default='./data/real_A/',help='Real images with highlight used to comparison')
    parser.add_argument('--result_path',default='./kid_score/',help='Path used to save KID calculation results')
    parser.add_argument('--normal',type=bool,default=True,help='Whether to input normalized image')
    parser.add_argument('--img_num',default=[870,1200],help='Number of images used to caculate,[real,fake]')
    parser.add_argument('--subset_size',type=int,default=50)
    parser.add_argument('--epoch_num',type=int,default=40,help='Number of epochs')
    parser.add_argument('--epoch_int',type=int,default=5,help='Interval of epochs')
    parser.add_argument('--observe_num',type=int,default=5,help='Number of epochs used to observe')
    args = parser.parse_args()
    return args

def image2tensor(root,start,end):
    data_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
    imgFolder = datasets.ImageFolder(root, transform=data_transform)
    imgLoader = torch.utils.data.DataLoader(imgFolder)
    count=0
    for X,_ in imgLoader:
        if count==start:
            img=X
        elif count>start:
            img=torch.cat((img,X),0)
        if count==end-1:
            return img if args.normal else img*255
        count+=1

def save(kid_mean,kid_std):
    dataframe = pd.DataFrame({'kid_mean':kid_mean,'kid_std':kid_std})
    dataframe.to_csv(args.result_path+'result.csv',index=False,sep=',')

def min_num(input,num):
    t = copy.deepcopy(input)
    min_number = []
    min_index = []
    for _ in range(num):
        number = min(t)
        index = t.index(number)
        t[index] = 1e10
        min_number.append(number)
        min_index.append(index*args.epoch_int+args.epoch_int)
    t = []
    print(f'the {num} lowest epochs:')
    print(f'index:{min_index}')
    print(f'kid score:{min_number}')
    return min_number,min_index

def splicing(images_path,image_row,image_column,image_int,image_size,image_save_path,image_name):
    #拼图
    IMAGES_FORMAT = ['.png','.PNG','.bmp','.BMP','jpg','JPG'] 
    image_names = [name for name in os.listdir(images_path) for item in IMAGES_FORMAT if
            os.path.splitext(name)[1] == item]
    if image_int!=0:
        image_names=image_names[::image_int]

    if len(image_names) != image_row * image_column:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    def image_compose():
        to_image = Image.new('RGB', (image_column * image_size[0], image_row * image_size[1])) 
        for y in range(1, image_row + 1):
            for x in range(1, image_column + 1):
                from_image = Image.open(images_path + image_names[image_column * (y - 1) + x - 1]).resize(
                    (image_size[0], image_size[1]),Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * image_size[0], (y - 1) * image_size[1]))
        return to_image.save(image_save_path+image_name+'.png')
    image_compose()

def plot(kid_mean,min_number,min_index):
    plt.plot(list(range(args.epoch_int,args.epoch_num*args.epoch_int+1,args.epoch_int)),kid_mean,'b-',label='mean')
    plt.title('KID score')
    plt.xlabel('epoch')
    plt.ylabel('kid')
    plt.grid()
    plt.legend()
    plt.scatter([min_index], [min_number], s=25, c='r',marker='x')  # stroke, colour
    plt.savefig(args.result_path+'result.png')
    return 0

def main(args):
    _ = torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    CLIP,_ = clip.load("ViT-B/32", device=device)

    real=image2tensor(args.real_path,0,args.img_num[0]).to(device)#加载real_img
    with torch.no_grad():
        real_features= CLIP.encode_image(real).to(device)#计算real_img的feature

    kid_mean,kid_std=[],[]

    for i in range(args.epoch_int,args.epoch_num*args.epoch_int+1,args.epoch_int):

        fake=image2tensor((args.fake_path+'test_{}').format(i),0,args.img_num[1]).to(device)#分别加载fake_img
        with torch.no_grad():
            fake_features= CLIP.encode_image(fake).to(device)#计算fake_img的feature

        KID = KernelInceptionDistance(subset_size=args.subset_size)#调用LightningAI的KID计算方法
        KID.real_features=real_features 
        KID.fake_features=fake_features
        mean_temp,std_temp = KID.compute()

        kid_mean.append(mean_temp)#方便保存数据
        kid_std.append(std_temp)

        kid_mean=torch.tensor(kid_mean, device="cpu").to(torch.float32).numpy().tolist()
        kid_std=torch.tensor(kid_std, device="cpu").to(torch.float32).numpy().tolist()

        print(f'epoch{i},kid_mean:{float(mean_temp):f},',f'kid_std:{float(std_temp):f}')

    save(kid_mean,kid_std)#保存结果
    min_number,min_index=min_num(kid_mean,args.observe_num)#计算kid_mean最低的几个epoch
    plot(kid_mean,min_number,min_index)#画图

    #拼图
    for index in min_index:
        splicing(args.fake_path+f'test_{index}/images/',int(args.img_num[1]/5),1,5,[256,256],args.result_path+'cache/','view_'+str(index))
        #其中5表示隔5张取一张图片放到大图中观察
    splicing(args.real_A_path+'images/',int(args.img_num[1]/5),1,5,[256,256],args.result_path+'cache/','view_0')
    splicing(args.result_path+'cache/',1,args.observe_num+1,0,[256,256*int(args.img_num[1]/5)],args.result_path,'view')

if __name__ == '__main__':
    args = get_args()
    main(args)