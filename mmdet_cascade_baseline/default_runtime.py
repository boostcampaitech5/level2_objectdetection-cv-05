checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=500),
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project='01km',
                entity = 'boostcamp-cv5',
                name = 'test'
            ),
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]
# 1 epoch에 train과 validation을 모두 하고 싶으면 workflow = [('train', 1), ('val', 1)]