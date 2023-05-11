# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001) ## lr 줄임 
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step', # 어떤 scheduler 를 쓸건지
    warmup='linear', # warmup을 할건지
    warmup_iters=500, # warmup iteration 얼마나 줄건지
    warmup_ratio=0.001, 
    step=[8, 11]) # step은 얼마마다 밟은 건지
runner = dict(type='EpochBasedRunner', max_epochs=12)