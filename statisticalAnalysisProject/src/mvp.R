library(rJava)
library(xlsx)
library(ggplot2)
library(treemapify)
library(gganimate)
library(tweenr)
library(RColorBrewer)
library(magrittr)
library(dplyr)
library(stringr)
library(plyr)
library(grid)
library(Cairo)
library(lattice)
rm(list=ls())
# ============================================================
# ====== 读取数据 ============================================
# ============================================================
source('C:/Users/Franc/Desktop/Dir/Rstudio/CourseR/mytheme.R')
basic <- read.csv('C:/Users/Franc/Desktop/Cur/# Course/Statistics Analysis/Data/basic_data.csv',header=T, stringsAsFactors = FALSE)
finan <- read.csv('C:/Users/Franc/Desktop/Cur/# Course/Statistics Analysis/Data/finance_data.csv',header=T, stringsAsFactors = FALSE)
com <- read.xlsx('C:/Users/Franc/Desktop/Cur/# Course/Statistics Analysis/Data/Complement.xlsx',sheetIndex = 2,header=T, stringsAsFactors = FALSE)[,1:2]%>%set_colnames(c('年份','定增'))

# ============================================================
# ====== 数据处理 ============================================
# ============================================================
industry <- basic[,c(1:3,5,15)]
marketvalue <- finan[,c(1:4,6,10,14:16)]
data1 <- plyr::join(marketvalue, industry, by='证券代码' ,type = 'full')
data1_agg <- ddply(data1, .(年份,一级行业), summarise,
                   company_num = length(证券代码),
                   mvtotal = sum(总市值1, na.rm = T),
                   nptotal = sum(净利润, na.rm = T),
                   assettotal = sum(资产总计, na.rm = T))
data1_agg$area <- rep(NA, nrow(data1_agg))
data1_agg[data1_agg$一级行业 %in% c('银行','非银金融'),]$area <- '金融'
data1_agg[data1_agg$一级行业 %in% c('建筑装饰','建筑材料'),]$area <- '建筑'
data1_agg[data1_agg$一级行业 %in% c('传媒','商业贸易','休闲服务'),]$area <- '服务'
data1_agg[data1_agg$一级行业 %in% c('钢铁','有色金属'),]$area <- '金属'
data1_agg[data1_agg$一级行业 %in% c('家用电器','食品饮料',
                                '轻工制造','纺织服装'),]$area <- '轻制造'
data1_agg[data1_agg$一级行业 %in% c('采掘','化工'),]$area <- '化工'
data1_agg[data1_agg$一级行业 %in% c('电子','通信','计算机'),]$area <- '电子设备'
data1_agg[data1_agg$一级行业 %in% c('汽车','电气设备','机械设备'),]$area <- '制造'
data1_agg[data1_agg$一级行业 %in% c('医药生物'),]$area <- '医药'
data1_agg[data1_agg$一级行业 %in% c('农林牧渔'),]$area <- '农业'
data1_agg[data1_agg$一级行业 %in% c('综合','交通运输','国防军工',
                                '公用事业','房地产'),]$area <- '公共'
data1_all <- ddply(data1_agg, .(年份,area), summarise,
                   company = sum(company_num, na.rm = T),
                   mv = sum(mvtotal, na.rm = T),
                   np = sum(nptotal, na.rm = T),
                   asset = sum(assettotal, na.rm = T))

# ============================================================
# ====== 树状图 ==============================================
# ============================================================
ggplot(data1_all[data1_all$年份 %in% c(2008,2017),],aes(area=mv,label=area))+
  geom_treemap(aes(fill= company),color='white')+
  geom_treemap_text(fontface='italic',size=15,colour='black',
                    place='topleft',reflow=T,alpha=0.9)+
  scale_fill_distiller('',palette='Blues',direction=1)+guides(fill=FALSE)+
  #labs(title='各行业市值分布',
  #     captions='注:格子面积与行业市值正比,颜色深度与行业企业数正比')+
  facet_grid(~年份)

# ============================================================
# ====== 扇形图 ==============================================
# ============================================================
data1_all$area <- factor(data1_all$area, 
                         levels= data1_all[order(-data1_all[data1_all$年份==2017,]$mv),]$area,
                         labels = data1_all[order(-data1_all[data1_all$年份==2017,]$mv),]$area,
                         ordered = TRUE)
data1_all <- data1_all[order(data1_all$年份,data1_all$area),]
data1_all2 <- ddply(data1_agg, .(年份), summarise,
                   com_all = sum(company_num, na.rm = T),
                   mv_all = sum(mvtotal, na.rm = T),
                   np_all = sum(nptotal, na.rm = T),
                   asset = sum(assettotal, na.rm = T))
data1_all <- data1_all%>%plyr::join(.,data1_all2, by = '年份')
data1_all$mv_per <- data1_all$mv/data1_all$mv_all
data1_all$np_per <- data1_all$np/data1_all$np_all
data1_agg <- data1_agg %>% plyr::join(., data1_all, by = c('年份','area'), type='full')


data_label1 <- data.frame(x=rep(2018,11), 
                         y = seq(1,0.04,length=11),
                         label = data1_all[data1_all$年份==2017,]$area,
                         angle= -(1-seq(1,0.04,length=11))*90)
data_label1
p1 <- 
  ggplot(data1_all)+
  geom_bar(aes(x=年份, y=mv_per, fill = area),stat = 'identity', position = 'stack')+
  geom_text(data=data_label1,aes(x=x, y=y, label = label, angle=angle),hjust=0,size=4,
            color='grey10')+
  guides(fill=FALSE)+
  xlim(1999,2018)+ylim(-3,1)+
  scale_fill_manual('', values=c(brewer.pal(9,'Blues')[c(6,5,3)],
                                 brewer.pal(9,'Greys')[c(3,6,5,4)],
                                 brewer.pal(11,'BrBG')[c(6,7,8,9)]))+
  coord_polar(theta = 'y',direction = -1)+theme_void()+
  theme(plot.margin = margin(c(0,0,0,0)),
        panel.spacing = margin(0.25,unit='pt'))
p3 <- 
  ggplot(data1_all2)+geom_bar(aes(年份,mv_all,fill=mv_all),color='white',stat='identity')+
  guides(fill=FALSE)+
  xlim(1978,2018)+#ylim()
  theme_void()+coord_polar(theta='x',start=pi/2+pi/40)+
  scale_fill_gradient2('',low=brewer.pal(9,'Greens')[9],
                       mid = brewer.pal(9,'Greys')[5],
                       high=brewer.pal(9,'Blues')[5]
                       )
p1 
CairoPNG(file="circletile1.png",width=2000,height=2000)
showtext_begin()
vie<-viewport(width=1.8,height=1.8,x=0.06,y=0.05)
vie3<-viewport(width=0.8,height=0.8,x=0.06,y=0.05)
vie2<-viewport(width=1.8,height=1.8,x=0.94,y=0.95)
vie4<-viewport(width=0.8,height=0.8,x=0.94,y=0.95)
grid.newpage()
print(p1,vp=vie)
print(p3,vp=vie3)
print(p2,vp=vie2)
print(p4,vp=vie4)
grid.text(label=2008:2017,x=rep(.035,11),y=seq(.40,.74,length=10),
          gp=gpar(col="grey10",fontsize=10,draw=TRUE,just="right"))
grid.text(label='2008总市值',x=0.035,y=0.15,rot=90,
          gp=gpar(col="grey10",fontsize=12,draw=TRUE,just="right"))
grid.text(label='Market Value',x=0.15,y=0.83333,
          gp=gpar(col="grey20",fontsize=24,draw=TRUE,just="right"))
grid.text(label=2008:2017,x=rep(.965,11),y=seq(.62,.28,length=10),
          gp=gpar(col="grey10",fontsize=10,draw=TRUE,just="left"))
grid.text(label=2008:2017,x=rep(.035,11),y=seq(.40,.74,length=10),
          gp=gpar(col="grey10",fontsize=10,draw=TRUE,just="right"))
grid.text(label='2008总利润',x=0.965,y=0.85,rot=90,
          gp=gpar(col="grey10",fontsize=12,draw=TRUE,just="right"))
grid.text(label='Profit',x=0.88,y=0.16666,
          gp=gpar(col="grey10",fontsize=24,draw=TRUE,just="left"))
showtext_end()
dev.off()
library(scales)
# c(brewer.pal(9,'Greys')[3],brewer.pal(9,'GnBu')[c(4,5,6,7,8)])
show_col(c(brewer.pal(9,'Greys')[3],brewer.pal(9,'GnBu')[c(4,5,6,7,8)]))
# ============================================================
# ====== 扇形图 ==============================================
# ============================================================
data1_all$area <- factor(data1_all$area, 
                         levels= data1_all[order(-data1_all[data1_all$年份==2017,]$np),]$area,
                         labels = data1_all[order(-data1_all[data1_all$年份==2017,]$np),]$area,
                         ordered = TRUE)
data1_all <- data1_all[order(data1_all$年份,data1_all$area),]
data_label2 <- data.frame(x=rep(2018,11), 
                          y = seq(1,0.04,length=11),
                          label = data1_all[data1_all$年份==2017,]$area,
                          angle= (1-seq(1,0.04,length=11))*90)

p2 <- 
  ggplot(data1_all)+
  geom_bar(aes(x=年份, y=np_per, fill = area),stat = 'identity', position = 'stack')+
  #geom_text(aes(x=年份,y=0, label=年份))+
  geom_text(data=data_label2,aes(x=x, y=y, label = label, angle=-angle),hjust=1,size=4,
            color='grey10')+
  xlim(1999,2018)+ylim(-2.9,1.1)+
  guides(fill=FALSE)+
  scale_fill_manual('', values=c(brewer.pal(9,'Blues')[c(6,5,3)],
                                 brewer.pal(9,'Greys')[c(3,6,5,4)],
                                 brewer.pal(11,'BrBG')[c(6,7,8,9)]))+
  coord_flip()+
  coord_polar(theta = 'y',direction = -1, start=-pi/2*1.9)+theme_void()+
  theme(plot.margin = margin(c(0,0,0,0)),
        panel.spacing = margin(0.25,unit='pt'))
p2
p4 <- 
  ggplot(data1_all2)+geom_bar(aes(年份,np_all,fill=np_all),color='white',stat='identity')+
  guides(fill=FALSE)+
  xlim(1978,2018)+#ylim()
  theme_void()+coord_polar(theta='x',start=-pi/2+pi/40)+
  scale_fill_gradient2('',low=brewer.pal(9,'Greens')[9],
                       mid = brewer.pal(9,'Greys')[5],
                       high=brewer.pal(9,'Blues')[5]
  )
p4


