library(readxl)
library(stringr)
library(magrittr)
library(recharts)
library(reshape2)
library(dplyr)
data <- read_excel('information.xlsx',sheet = "s")

##############
##### finance_data
#############
finance <- data[,c(1:2,157,3:102,104:123,129:148)]

finance_long <- reshape2::melt(finance,id.vars=c("证券代码↑","证券简称","省份"))
l <- str_split(finance_long$variable,'\r\n\r\n')

finance_long$v1 <- sapply(l,FUN = function(x) x[2]) %>% str_extract_all(pattern = '\\d{4}') %>% as.integer()

finance_long$v2 <- sapply(l,FUN = function(x) x[1]) 
finance_long$v3 <- sapply(l,FUN = function(x) x[length(x)]) %>%str_replace_all('\\[单位\\] ','')
finance_data <- finance_long[,c(1,2,3,6,7,8,5)]
colnames(finance_data) <- c("证券代码","证券简称","省份","年份","科目","单位","数值")
finance_data <- dcast(finance_data,`证券代码`+`证券简称`+`省份`+`年份`~`科目`,value.var = '数值',mean)
finance_data <- finance_data[order(finance_data$年份),]
finance_data$省份 <- str_replace_all(finance_data$省份,"省","")
finance_data$省份[finance_data$省份=="新疆维吾尔自治区"] <- "新疆"
finance_data$省份[finance_data$省份=="内蒙古自治区"] <- "内蒙古"
finance_data$省份[finance_data$省份=="广西壮族自治区"] <- "广西"
finance_data$省份[finance_data$省份=="宁夏回族自治区"] <- "宁夏"
finance_data$省份[finance_data$省份=="西藏自治区"] <- "西藏"


##############
##### basic_data
#############


basic <- data[,c(1,2,124:128,149:174)]
basic_data <- basic[,c(1,2,3,4,5,7:12,15,16,17,28:33)]
colnames(basic_data) <- c("证券代码","证券简称","上市日期","价格(元)","募集资金(百万元)","区间增发募集资金","上市板","上市地点","戴帽摘帽时间","证券曾用名","概念板块","成立日期","省份","城市","一级行业","明细行业","博士占比","硕士占比","本科占比","专科占比")
basic_data$上市日期 <- as.POSIXlt(basic_data$上市日期)$year+1900
basic_data$省份 <- str_replace_all(basic_data$省份,"省","")
basic_data$省份[basic_data$省份=="新疆维吾尔自治区"] <- "新疆"
basic_data$省份[basic_data$省份=="内蒙古自治区"] <- "内蒙古"
basic_data$省份[basic_data$省份=="广西壮族自治区"] <- "广西"
basic_data$省份[basic_data$省份=="宁夏回族自治区"] <- "宁夏"
basic_data$省份[basic_data$省份=="西藏自治区"] <- "西藏"





write.csv(finance_data,file ="finance_data.csv",fileEncoding = 'gbk',row.names = F)
write.csv(basic_data,file ="basic_data.csv",fileEncoding = 'gbk',row.names = F)


##############
##### 第一张交互地图-A股三十年
#############

map_data_tmp <- aggregate(basic_data$证券代码,by=list('year'=basic_data$上市日期,'prov'=basic_data$省份),length)
map_data_tmp <- map_data_tmp[order(map_data_tmp$year),]
colnames(map_data_tmp)[3] <- 'count'

d_tmp <- data.frame(
'year'=rep(1990:2018,each=length(unique(basic_data$省份))),
'prov'=rep(unique(basic_data$省份),length(1990:2018))
)
map_data <- merge(d_tmp,map_data_tmp,by.x = c('year','prov'),by.y = c('year','prov'),all.x=T)
map_data$count[is.na(map_data$count)] <- 0

save(map_data,file="map_data.rdata")

splitList <-list(
    list(start=1, end=1, label='1', color='#B0C4DE'),
    list(start=2, end=2, label='2', color='#87CEFA'),
    list(start=3, end=3, label='3', color='#00BFFF'),
    list(start=4, end=6, label='4-6', color='#4682B4'),
    list(start=7, end=11, label='7-11', color='#0000CD'),
    list(start=12, end=98, label='>=12', color='#00008B')
)    

echartr(map_data, prov, count, t=year, type="map_china") %>%
  setDataRange(splitList=splitList,pos=3) %>%
  #setTimeline(y=50,autoPlay = T,playInterval=300) %>%
  setTimeline(playInterval=300,height ="30px",loop = F) %>%
  #setToolbox(pos=3) %>%
  setToolbox(show = F) %>%
  setSeries(showLegendSymbol=F) %>%
  setLegend(show=F) %>%
  setTitle('上市公司数',pos=12)



##############
##### 第二张交互地图-A股省际分布市值
#############

GDP <- read.csv(file='GDP.csv',stringsAsFactors = F)

capital_data <- aggregate(finance_data$总市值1,by=list('year'=finance_data$年份,'prov'=finance_data$省份),sum,na.rm=T)

GDP_capital_data <- left_join(GDP,capital_data,by=c('地区'='prov','年份'='year')) %>% left_join(map_data,by=c('地区'='prov','年份'='year')) %>% suppressWarnings()
colnames(GDP_capital_data) <- c('prov','year','GDP','capital','count')

GDP_data <- subset(GDP_capital_data,year>2007  & prov %in% c('安徽','北京','福建','广东','河南','湖南','江苏','山东','上海','浙江'))
GDP_data <- dplyr::arrange(GDP_data,prov,year)


#vec <- NULL
#for (i in 1:10) {
#  vec <- c(vec,cumsum(GDP_data$count[(9*i-8):(9*i)]))
#}
#GDP_data$sum <- vec
save(GDP_data,file="GDP_data.rdata")


#GDP_data <- GDP_data[GDP_data$prov %in% c('北京','福建','广东','上海','浙江'),]

echartr(GDP_data, capital,GDP,t=year,series = prov,type='bubble') %>% 
  #c('circle', 'rectangle', 'triangle', 'diamond', 'emptyCircle', 'emptyRectangle', 'emptyTriangle', 'emptyDiamond')
  setSeries(series = '福建',symbol='star',itemStyle=list(normal=itemStyle(color='red')),symbolSize=20) %>%
  setSeries(series = '北京',symbol='diamond',symbolSize=20) %>%
  setSeries(series = '上海',symbol='diamond',symbolSize=20) %>%
  setSeries(series = '广东',symbol='diamond',symbolSize=20) %>%
  setSeries(series = '浙江',symbol='circle',symbolSize=20) %>%
  setSeries(series = '山东',symbol='circle',symbolSize=20) %>%
  setSeries(series = '江苏',symbol='circle',symbolSize=20) %>%
  setSeries(series = '河南',symbol='circle',symbolSize=20) %>%
  setSeries(series = '安徽',symbol='circle',symbolSize=20) %>%
  setSeries(series = '湖南',symbol='circle',symbolSize=20) %>%
  setXAxis(name='总市值(亿)',max=1.05*max(GDP_data$capital),min=0,axisLabel=list(formatter="%d")) %>%
  setYAxis(name='GDP(亿)',max=1.05*max(GDP_data$GDP),min=0) %>%
  setToolbox(show = F) %>%
  setLegend(pos = 3,itemGap = 20) %>%
  setTitle('GDP-总市值省际分布',pos=12) %>%
  setTimeline(playInterval=350)








##############
#####  仪表盘交互图-A股板块热度
#############

finance_data1 <- left_join(finance_data,basic_data[,c('证券代码' ,'上市板')],by='证券代码')
gauge_data <- aggregate(finance_data1$区间涨跌幅,by=list('year'=finance_data1$年份,'ban'=finance_data1$上市板),function(x) 
  round(sum(na.omit(x)>0)/length(na.omit(x)),2)*100)

gauge_data <- gauge_data[gauge_data$year!=2008,]
colnames(gauge_data) <- c('year','ban','percent')
gauge_data$ban[gauge_data$ban=="中小企业板"] <- "中小板"
gauge_data$ban <- paste0(gauge_data$ban,"(%)")
save(gauge_data,file='gauge.rdata')

echartr(gauge_data, ban, percent, facet=ban,type='gauge',t=year)%>% 
  setTitle('板块热度仪表板',pos=12) %>%
  setToolbox(show = F) %>%
  setSeries(legendHoverLink=F,splitNumber=4,
            title=list(show=TRUE, offsetCenter=list(0, '120%'),
            textStyle=list(color='#333', fontSize=25)),
            axisLabel=list(show=TRUE,offsetCenter=list(5, '120%') ),
            axisTick=list( show=TRUE, splitNumber=5, length=6,lineStyle=list( color='#eee', width=1, type='solid')),
            axisLine=list( show=TRUE, lineStyle=list(color=list(list(0.2, '#228b22'), list(0.8, '#48b'), list(1, '#ff4500')), width=20)),
            splitLine=list(show=TRUE, length=20, lineStyle=list( color='#eee', width=2, type='solid')),
            axisLabel=list(show=TRUE, formatter='', textStyle=list(color='auto')),
            detail=list(show=TRUE, backgroundColor='rgba(0,0,0,0)', borderWidth=0, borderColor='#ccc', width=80, height=40, offsetCenter=list(0, '40%'), formatter=NULL, textStyle=list(color='auto', fontSize=30))
            ) %>%
  setLegend(show=F) 
  
  




finance_data2 <- left_join(finance_data,basic_data[,c('证券代码' ,'上市板','城市','一级行业','上市日期')],by='证券代码')
finance_data2_cal <- finance_data2[finance_data2$年份!=finance_data2$上市日期,c('年份','证券简称','上市日期','区间涨跌幅','一级行业','城市','上市板','总市值1','净利润(同比增长率)')]

#####每年的前十涨幅
d1 <- dplyr::arrange(finance_data2_cal,finance_data2_cal$年份,desc(finance_data2_cal$区间涨跌幅))
d1_tmp <- data.frame()
for (i in 2008:2017) {
  d_tmp <- head(d1[d1$年份==i,],10)
  d1_tmp <- rbind(d1_tmp,d_tmp)
}


#####每年的前十降幅
d2 <- dplyr::arrange(finance_data2_cal,finance_data2_cal$年份,finance_data2_cal$区间涨跌幅)
d2_tmp <- data.frame()
for (i in 2008:2017) {
  d_tmp <- head(d2[d2$年份==i,],10)
  d2_tmp <- rbind(d2_tmp,d_tmp)
}

colnames(d1_tmp) <- c('年份','证券简称','上市年份','涨跌幅(%)','行业','城市','上市板','总市值(亿)','净利润增长率(%)')
colnames(d2_tmp) <- c('年份','证券简称','上市年份','涨跌幅(%)','行业','城市','上市板','总市值(亿)','净利润增长率(%)')

save(d1_tmp,file='d1_tmp.rdata')
save(d2_tmp,file='d2_tmp.rdata')
