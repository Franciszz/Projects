data_ipo <- ddply(basic, .(上市日期), summarise,
                  company = sum(募集资金.百万元., na.rm = T))[1:28,]%>%
  cbind('定增'=com$定增)%>%
  set_colnames(c('年份','IPO','定增'))
data_ipo$all <- data_ipo$IPO+data_ipo$定增
data_ipo$ipo_per <- data_ipo$IPO/data_ipo$all
data_ipo$nipo_per <- data_ipo$定增/data_ipo$all
library(reshape2)

data_ipomelt <- melt(data_ipo, id.vars = '年份', measure.vars = c('IPO','定增'))
ggplot(data_ipomelt)+geom_bar(aes(x=年份, y = value+1, fill = variable),
                              stat ='identity',position = 'stack')+
  scale_fill_manual('',values=c(brewer.pal(9,'Blues')[3],
                                brewer.pal(9,'Blues')[6]))+
  labs(x='年份',y='募集资金/百万')+
  theme_os()


data_ipo$all <- log(data_ipo$all)
data_ipo$ipo_tran <- data_ipo$all*data_ipo$ipo_per
year_index <- c(1990:2017,2017:1990)
all_index <- c(data_ipo$all,data_ipo$ipo_tran[28:1])
ipo_index <- c(data_ipo$ipo_tran, rep(0,28))
dat <- data.frame(year = c(year_index,year_index),
                  all = c(all_index, ipo_index),
                  type = c(rep('all',56),rep('ipo',56)))

ggplot(dat)+
  geom_polygon(aes(x=year,y=all,fill=type))+
  guides(fill=FALSE)+
  scale_fill_manual('',values=c(brewer.pal(9,'Blues')[6],
                                brewer.pal(9,'Greys')[3]))+
  annotate('text',x=2000,y=6.4,label='IPO融资额',size=8,color=brewer.pal(9,'Blues')[7])+
  annotate('text',x=2010,y=9.6,label='定增融资额',size=8,color=brewer.pal(9,'Greys')[2])+
  ylab('ln(融资额):百万')+xlab(NULL)+
  coord_equal(expand = c(0.1,0.1))+
  scale_x_discrete(limits=seq(1990,2017,2))+
  scale_y_log10(breaks=c(2,4,6,8,10,12))+theme_os()

data_ipo$all <- data_ipo$all/2
center_x <- c(4,10,18,28,38)
mon_all <- data_ipo$all[19:28]/data_ipo$all[19]
x_list <- c()
y_list <- c()
index_list <- c()
for (i in 1:length(center_x)){
  px = seq(from=center_x[i]-mon_all[i],to=center_x[i]+mon_all[i],length=1000)
  py = sqrt(1e-10+(mon_all[i])^2-(px-center_x[i])^2)
  x_list = c(x_list, px, rev(px))
  y_list = c(y_list, py-mon_all[i]+6, -py-mon_all[i]+6)
  index_list = c(index_list, rep(i+2007,2000))
}
for (i in 1:length(center_x)){
  px = seq(from=center_x[i]-mon_all[i+5],to=center_x[i]+mon_all[i+5],length=1000)
  py = sqrt(1e-10+(mon_all[i+5])^2-(px-center_x[i])^2)
  x_list = c(x_list, px, rev(px))
  y_list = c(y_list, py-12+mon_all[i+5], -py-12+mon_all[i+5])
  index_list = c(index_list, rep(i+2012,2000))
}
mycol <- brewer.pal(9,'Blues')
dat <- data.frame(x=x_list, y=y_list, year = index_list)

dat1 <- dat[dat$year==2008,]%>%filter(y<=(6-2*0.3436))
dat2 <- dat[dat$year==2009,]%>%filter(y<=(6-2*0.3886*1.487))
dat3 <- dat[dat$year==2010,]%>%filter(y<=(6-2*0.5893*2.7536))
dat4 <- dat[dat$year==2011,]%>%filter(y<=(6-2*0.4242*2.2000))
dat5 <- dat[dat$year==2012,]%>%filter(y<=(6-2*0.2350*1.4620))
dat6 <- dat[dat$year==2013,]%>%filter(y>=(-12+2*0*1.0594))
dat7 <- dat[dat$year==2014,]%>%filter(y>=(-12+2*0.1068*2.0725))
dat8 <- dat[dat$year==2015,]%>%filter(y>=(-12+2*0.2*3.6048))
dat9 <- dat[dat$year==2016,]%>%filter(y>=(-12+2*0.1024*4.8555))
dat10 <- dat[dat$year==2017,]%>%filter(y>=(-12+2*0.25*2.8234))

ggplot(dat)+
  geom_hline(yintercept = 6.2,col='black',size=1)+
  geom_hline(yintercept = -12.2,col='black',size=1)+
  geom_polygon(data=dat1,aes(x,y,group=year),fill=mycol[4],col=NA)+
  geom_polygon(data=dat2,aes(x,y,group=year),fill=mycol[4],col=NA)+
  geom_polygon(data=dat3,aes(x,y,group=year),fill=mycol[2],col=NA)+
  geom_polygon(data=dat4,aes(x,y,group=year),fill=mycol[4],col=NA)+
  geom_polygon(data=dat5,aes(x,y,group=year),fill=mycol[5],col=NA)+
  geom_polygon(data=dat6,aes(x,y,group=year),fill=mycol[8],col=NA)+
  geom_polygon(data=dat7,aes(x,y,group=year),fill=mycol[7],col=NA)+
  geom_polygon(data=dat8,aes(x,y,group=year),fill=mycol[6],col=NA)+
  geom_polygon(data=dat9,aes(x,y,group=year),fill=mycol[7],col=NA)+
  geom_polygon(data=dat10,aes(x,y,group=year),fill=mycol[6],col=NA)+
  geom_path(aes(x=x, y=y, group=year),size=0.5,col='black')+
  annotate('text',x=center_x,y=7.2,label=2008:2012, size=4)+
  annotate('text',x=center_x,y=-13.2,label=2013:2017, size=4)+
  annotate('text',x=18,y=4,label='IPO',size=4)+
  annotate('text',x=18,y=2,label='增发', size=4)+
  ylim(-18,13)+theme_void()


# px1<-seq(from=3.6-data_ipo$all[1],to=3.6+data_ipo$all[1],length=1000)
# py1<-sqrt(0.0000001+(data_ipo$all[1])^2-(px1-3.6)^2)
# po1_x <- c(px1,rev(px1))
# po1_y <- c(py1,-py1)+5
# px2<-seq(from=center_x[2]-data_ipo$all[2],to=center_x[2]+data_ipo$all[2],length=1000)
# py2<-sqrt(0.0000001+(data_ipo$all[2])^2-(px2-center_x[2])^2)
# po2_x <- c(px2,rev(px2))
# po2_y <- c(py2,-py2)+5
# px3<-seq(from=center_x[3]-data_ipo$all[3],to=center_x[3]+data_ipo$all[3],length=1000)
# py3<-sqrt(0.0000001+(data_ipo$all[3])^2-(px3-center_x[3])^2)
# po3_x <- c(px3,rev(px3))
# po3_y <- c(py3,-py3)+5
# px4<-seq(from=center_x[4]-data_ipo$all[4],to=center_x[4]+data_ipo$all[4],length=1000)
# py4<-sqrt(0.0000001+(data_ipo$all[4])^2-(px4-center_x[4])^2)
# po4_x <- c(px4,rev(px4))
# po4_y <- c(py4,-py4)+5
# px5<-seq(from=center_x[5]-data_ipo$all[5],to=center_x[5]+data_ipo$all[5],length=1000)
# py5<-sqrt(0.0000001+(data_ipo$all[5])^2-(px5-center_x[5])^2)
# po5_x <- c(px5,rev(px5))
# po5_y <- c(py5,-py5)+5
# px6<-seq(from=3.6-data_ipo$all[6],to=3.6+data_ipo$all[6],length=1000)
# py6<-sqrt(0.0000001+(data_ipo$all[6])^2-(px6-3.6)^2)
# po6_x <- c(px6,rev(px6))
# po6_y <- c(py6,-py6)-5
# px7<-seq(from=center_x[7]-data_ipo$all[7],to=center_x[7]+data_ipo$all[7],length=1000)
# py7<-sqrt(0.0000001+(data_ipo$all[7])^2-(px7-center_x[7])^2)
# po7_x <- c(px7,rev(px7))
# po7_y <- c(py7,-py7)-5
# px8<-seq(from=center_x[8]-data_ipo$all[8],to=center_x[8]+data_ipo$all[8],length=1000)
# py8<-sqrt(0.0000001+(data_ipo$all[8])^2-(px8-center_x[8])^2)
# po8_x <- c(px8,rev(px8))-5
# po8_y <- c(py8,-py8)
# px9<-seq(from=center_x[9]-data_ipo$all[9],to=center_x[9]+data_ipo$all[9],length=1000)
# py9<-sqrt(0.0000001+(data_ipo$all[9])^2-(px9-center_x[9])^2)
# po9_x <- c(px9,rev(px9))-5
# po9_y <- c(py9,-py9)
# px10<-seq(from=center_x[10]-data_ipo$all[10],to=center_x[10]+data_ipo$all[10],length=1000)
# py10<-sqrt(0.0000001+(data_ipo$all[10])^2-(px10-center_x[10])^2)
# po10_x <- c(px10,rev(px10))
# po10_y <- c(py10,-py10)-5
# ggplot()+
#   geom_path(aes(x=po1_x,y=po1_y),color='skyblue')+
#   geom_path(aes(x=po2_x,y=po2_y),color='skyblue')+
#   geom_path(aes(x=po3_x,y=po3_y),color='skyblue')+
#   geom_path(aes(x=po4_x,y=po4_y),color='skyblue')+
#   geom_path(aes(x=po5_x,y=po5_y),color='skyblue')+
#   geom_path(aes(x=po6_x,y=po6_y),color='skyblue')+
#   geom_path(aes(x=po7_x,y=po7_y),color='skyblue')+
#   geom_path(aes(x=po8_x,y=po8_y),color='skyblue')+
#   geom_path(aes(x=po9_x,y=po9_y),color='skyblue')+
#   geom_path(aes(x=po10_x,y=po10_y),color='skyblue')+
#   ylim(-12,12)+
#   theme_void()
