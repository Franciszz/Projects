ggplot(data1_all[data1_all$年份 %in% c(2008,2017),],aes(area=mv,label=area))+
  geom_treemap(aes(fill= company),color='white')+
  geom_treemap_text(fontface='italic',size=15,colour='black',
                    place='topleft',reflow=T,alpha=0.9)+
  scale_fill_distiller('',palette='Blues',direction=1)+guides(fill=FALSE)+
  labs(title='各行业市值分布',
       captions='注:格子面积与行业市值正比,颜色深度与行业企业数正比')+
  facet_grid(~年份)