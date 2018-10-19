finance_data1 <- left_join(finance_data,basic_data[,c('证券代码' ,'上市板')],by='证券代码')
aov_data <- finance_data1[,c('年份','区间涨跌幅','上市板')]
colnames(aov_data) <- c( 'year','updown','ban')
aov_data <- na.omit(aov_data)
aov_data$updown <- ifelse(aov_data$updown>0,1,0)



aov_result <- list()
for (year in 2008:2017) {
  d <- aov_data[aov_data$year==year,]
  aov_result[as.character(year)]=summary(aov(updown~ban,d))
}

save(aov_result,file='aov_result.rdata')

aov_result_data <- data.frame('year'=2008:2017)
aov_result_data$Df1 <- sapply(aov_result,function(x) x$`Df`[1])
aov_result_data$Df2 <- sapply(aov_result,function(x) x$`Df`[2])

aov_result_data$f <- sapply(aov_result,function(x) x$`F value`[1]) %>% round(2)
aov_result_data$p <- sapply(aov_result,function(x) x$`Pr(>F)`[1]) %>% format(scientific=TRUE,digit=2)

save(aov_result_data,file='aov_result_data.rdata')

