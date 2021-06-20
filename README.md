# self-imitation-scikit-learn

This is an attempt at using scikit-learn for behavioral cloning to build RL algorithms using the self-imitation learning approach.

We can use self-imitation learning with behavioral cloning in order to tackle the cartpole task without using deep learning frameworks

```

                                             Rollout Return                                         
     ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
199.0┤                                                                                      ▖▘▘▝▝▝▝│
178.0┤                                                                                ▗▗ ▝ ▘       │
     │                                                                               ▖  ▝          │
157.0┤                                                                          ▗▝ ▖▘              │
     │                                                                      ▖▖▝▝                   │
     │                                                                  ▗ ▖▖                       │
137.0┤                                                            ▖ ▖▗▗▝                           │
116.0┤                                                        ▗▝▝  ▘                               │
     │                                            ▖ ▗▗▗▝ ▘▘▘▘▝                                     │
 95.0┤                                     ▖▗▗▗▝ ▘ ▘                                               │
     │                             ▖▗▗▗ ▘▘▘                                                        │
 74.0┤▖▘▘▘▝▝▗▝ ▖▖▖▖▗▗▗ ▖▖▘▖▝▝▝▝ ▘▘▘                                                                │
     └┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬┘
      0                     20.0                    41                    62.0                    82

82: average return 198.99
```

