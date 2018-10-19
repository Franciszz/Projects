library(plyr)
library(magrittr)
library(randomForest)
library(Matrix)
library(mice)
library(stringr)
basic2 <- basic[,c(2,7,15)]
fin2 <- finan[,c(2:5,7:10,12:14,16:18)]
data2 <- plyr::join(basic2,fin2,by='证券简称')%>%.[!is.na(.$区间涨跌幅),]
data2$area <- rep(NA, nrow(data2))
data2[data2$一级行业 %in% c('银行','非银金融'),]$area <- '金融'
data2[data2$一级行业 %in% c('建筑装饰','建筑材料'),]$area <- '建筑'
data2[data2$一级行业 %in% c('传媒','商业贸易','休闲服务'),]$area <- '服务'
data2[data2$一级行业 %in% c('钢铁','有色金属'),]$area <- '金属'
data2[data2$一级行业 %in% c('家用电器','食品饮料',
                                '轻工制造','纺织服装'),]$area <- '轻制造'
data2[data2$一级行业 %in% c('采掘','化工'),]$area <- '化工'
data2[data2$一级行业 %in% c('电子','通信','计算机'),]$area <- '电子设备'
data2[data2$一级行业 %in% c('汽车','电气设备','机械设备'),]$area <- '制造'
data2[data2$一级行业 %in% c('医药生物'),]$area <- '医药'
data2[data2$一级行业 %in% c('农林牧渔'),]$area <- '农业'
data2[data2$一级行业 %in% c('综合','交通运输','国防军工',
                                '公用事业','房地产'),]$area <- '公共'
temdata2 <- mice(data2,m=4,maxit=10,meth='pmm',seed=500)
temdata2 <- complete(temdata2,1)
temdata2$一级行业 <- data2$area
row.names(temdata2) <- 1:nrow(temdata2)
temdata3 <- sparse.model.matrix(~ 上市板+一级行业+省份, data=temdata2)
temdata_com <- as.data.frame(as.array(temdata3))[,2:43]
a <- names(temdata_com)
names(temdata_com) <- c(str_sub(a[1:2],4,-1),str_sub(a[3:12],5,-1),str_sub(a[13:42],3,-1))
data3 <- cbind(temdata2,temdata_com)
data_model <- data3[,c(1,10,6:9,11:16,17:58)]
data_model[,3:12] <- scale(data_model[,3:12],center=TRUE,scale=TRUE)
write.csv(data_model,file='C:/Users/Franc/Desktop/Cur/# Course/Statistics Analysis/Data/rnf_data.csv')
# =======================================================================
# RandomForest
# =======================================================================
names(data_model)
rnf_reg <- randomForest(区间涨跌幅~.,data=data_model[,2:length(data_model)],
                  ntree=200,importance=TRUE)
imp <- importance(rnf_reg,type=1)

varImpPlot(rnf_reg)
class(importance)

# =======================================================================
# Variable Importance
# =======================================================================
library(rJava)
library(xlsx)
var_imp <- read.xlsx('C:/Users/Franc/Desktop/Cur/# Course/Statistics Analysis/Data/var_imp.xlsx',
                     sheetIndex = 1,header=T,encoding = 'utf8')
var_imp <- data.frame(imp = var_imp$imp,var = names(data_model)[c(3:12,14:54)])
var_imp <- var_imp[order(-var_imp$imp),]
var_imp$id <- 1:51
var_imp$label <- c('市值','ROE','净利润增长率','营收增长率','净经营现金',
                 '主营业务比例','资产负债率','ROA','每股股利','资产周转率',
                 as.character(var_imp$var[11:51]))
var_imp$type <- c(1,rep(2,9),rep(3,4),rep(4,37))
library(ggplot2)
library(RColorBrewer)

mycol <- c(brewer.pal(9,'Greys')[3],brewer.pal(9,'GnBu')[c(4,5,6,7,8)])
ggplot(var_imp[1:15,])+
  geom_bar(aes(x=id, y = imp,fill=as.character(type)),stat='identity',width=0.8)+
  scale_x_reverse()+
  ylim(-0.06,0.26)+
  geom_text(aes(x=id,y=-0.02,label=label),hjust=1,size=4.2,vjust=0.3)+
  geom_text(aes(x=id,y=imp+0.018,label=paste0(round(imp*100,2),'%'),
                col=as.character(type)),size=4.2,vjust=0.3,fontface='bold')+
  scale_fill_manual('',values=brewer.pal(9,'Blues')[c(7,6,5,3)])+
  scale_color_manual('',values=brewer.pal(9,'Blues')[c(7,6,5,3)])+
  guides(fill=FALSE,color=FALSE)+
  coord_flip()+theme_void()
  
