
Before adding averaging:

INFO:pegasos:Pegasos starting, npoints=20242, ndims=47237
INFO:pegasos:Epoch 1 finished
INFO:hingeloss: loss: 0.225107 errors: 773 (3.819 percent)
INFO:pegasos:Epoch 2 finished
INFO:hingeloss: loss: 0.155259 errors: 412 (2.035 percent)
INFO:pegasos:Epoch 3 finished
INFO:hingeloss: loss: 0.159388 errors: 419 (2.070 percent)
INFO:pegasos:Epoch 4 finished
INFO:hingeloss: loss: 0.141698 errors: 330 (1.630 percent)
INFO:pegasos:Epoch 5 finished
INFO:hingeloss: loss: 0.142064 errors: 363 (1.793 percent)
INFO:pegasos:Epoch 6 finished
INFO:hingeloss: loss: 0.137616 errors: 336 (1.660 percent)
INFO:pegasos:Epoch 7 finished
INFO:hingeloss: loss: 0.137168 errors: 341 (1.685 percent)
INFO:pegasos:Epoch 8 finished
INFO:hingeloss: loss: 0.134976 errors: 326 (1.611 percent)
INFO:pegasos:Epoch 9 finished
INFO:hingeloss: loss: 0.135242 errors: 324 (1.601 percent)
INFO:pegasos:Epoch 10 finished
INFO:hingeloss: loss: 0.140550 errors: 345 (1.704 percent)
INFO:pegasos:Epoch 11 finished
INFO:hingeloss: loss: 0.135776 errors: 328 (1.620 percent)
INFO:pegasos:Epoch 12 finished
INFO:hingeloss: loss: 0.133005 errors: 315 (1.556 percent)
INFO:pegasos:Epoch 13 finished
INFO:hingeloss: loss: 0.132802 errors: 324 (1.601 percent)
INFO:pegasos:Epoch 14 finished
INFO:hingeloss: loss: 0.134600 errors: 318 (1.571 percent)
INFO:pegasos:Epoch 15 finished
INFO:hingeloss: loss: 0.134083 errors: 327 (1.615 percent)
INFO:pegasos:Epoch 16 finished
INFO:hingeloss: loss: 0.134155 errors: 330 (1.630 percent)
INFO:pegasos:Epoch 17 finished
INFO:hingeloss: loss: 0.133219 errors: 335 (1.655 percent)
INFO:pegasos:Epoch 18 finished
INFO:hingeloss: loss: 0.132289 errors: 329 (1.625 percent)
INFO:pegasos:Epoch 19 finished
INFO:hingeloss: loss: 0.131541 errors: 317 (1.566 percent)
INFO:pegasos:Epoch 20 finished
INFO:hingeloss: loss: 0.134781 errors: 313 (1.546 percent)


### Hinge loss, pointsaga:

INFO:hingeloss: loss: 1.000000 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:hingeloss: loss: 1.504339 errors: 7716 (38.119 percent)
INFO:pointsaga:Epoch 1 finished
INFO:hingeloss: loss: 0.265576 errors: 965 (4.767 percent)
INFO:pointsaga:Epoch 2 finished
INFO:hingeloss: loss: 0.157786 errors: 353 (1.744 percent)
INFO:pointsaga:Epoch 3 finished
INFO:hingeloss: loss: 0.209804 errors: 483 (2.386 percent)
INFO:pointsaga:Epoch 4 finished
INFO:hingeloss: loss: 0.149431 errors: 337 (1.665 percent)
INFO:pointsaga:Epoch 5 finished
INFO:hingeloss: loss: 0.135091 errors: 307 (1.517 percent)
INFO:pointsaga:Epoch 6 finished
INFO:hingeloss: loss: 0.133021 errors: 320 (1.581 percent)
INFO:pointsaga:Epoch 7 finished
INFO:hingeloss: loss: 0.131852 errors: 318 (1.571 percent)
INFO:pointsaga:Epoch 8 finished
INFO:hingeloss: loss: 0.131127 errors: 315 (1.556 percent)
INFO:pointsaga:Epoch 9 finished
INFO:hingeloss: loss: 0.132325 errors: 313 (1.546 percent)
INFO:pointsaga:Epoch 10 finished
INFO:hingeloss: loss: 0.131046 errors: 314 (1.551 percent)
INFO:pointsaga:Epoch 11 finished
INFO:hingeloss: loss: 0.129725 errors: 316 (1.561 percent)
INFO:pointsaga:Epoch 12 finished
INFO:hingeloss: loss: 0.129474 errors: 317 (1.566 percent)
INFO:pointsaga:Epoch 13 finished
INFO:hingeloss: loss: 0.129221 errors: 318 (1.571 percent)
INFO:pointsaga:Epoch 14 finished
INFO:hingeloss: loss: 0.129085 errors: 315 (1.556 percent)
INFO:pointsaga:Epoch 15 finished
INFO:hingeloss: loss: 0.128987 errors: 315 (1.556 percent)
INFO:pointsaga:Epoch 16 finished
INFO:hingeloss: loss: 0.128950 errors: 319 (1.576 percent)
INFO:pointsaga:Epoch 17 finished
INFO:hingeloss: loss: 0.128835 errors: 316 (1.561 percent)
INFO:pointsaga:Epoch 18 finished
INFO:hingeloss: loss: 0.129040 errors: 320 (1.581 percent)
INFO:pointsaga:Epoch 19 finished
INFO:hingeloss: loss: 0.128746 errors: 317 (1.566 percent)


Might have to tune step size:
0.1:
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.447723 errors: 4523 (22.345 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.275762 errors: 989 (4.886 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.289387 errors: 916 (4.525 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.299738 errors: 868 (4.288 percent)

0.01:
INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.549290 errors: 1924 (9.505 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.458742 errors: 1368 (6.758 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.401308 errors: 1292 (6.383 percent)

It seems to be converging I guess:

INFO:opt:Loading data
INFO:opt:Data with 20242 points and 47237 features loaded.
INFO:opt:Train Proportions: -1 9751   1: 10491
INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.604966 errors: 4938 (24.395 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.327357 errors: 745 (3.680 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.324319 errors: 720 (3.557 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.323311 errors: 746 (3.685 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.321049 errors: 709 (3.503 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.320656 errors: 709 (3.503 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.320424 errors: 711 (3.512 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.320515 errors: 710 (3.508 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.320341 errors: 709 (3.503 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.320383 errors: 710 (3.508 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.320383 errors: 708 (3.498 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.320393 errors: 710 (3.508 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.320394 errors: 710 (3.508 percent)

Lets try sgd with logistic loss:

INFO:pegasos:Pegasos starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pegasos:Epoch 1 finished
INFO:logisticloss:w: loss: 0.275913 errors: 716 (3.537 percent)
INFO:logisticloss:wbar: loss: 0.275651 errors: 617 (3.048 percent)
INFO:pegasos:Epoch 2 finished
INFO:logisticloss:w: loss: 0.263854 errors: 606 (2.994 percent)
INFO:logisticloss:wbar: loss: 0.265600 errors: 581 (2.870 percent)
INFO:pegasos:Epoch 3 finished
INFO:logisticloss:w: loss: 0.272412 errors: 758 (3.745 percent)
INFO:logisticloss:wbar: loss: 0.262604 errors: 577 (2.851 percent)
INFO:pegasos:Epoch 4 finished
INFO:logisticloss:w: loss: 0.260288 errors: 578 (2.855 percent)
INFO:logisticloss:wbar: loss: 0.261281 errors: 586 (2.895 percent)
INFO:pegasos:Epoch 5 finished
INFO:logisticloss:w: loss: 0.265373 errors: 666 (3.290 percent)
INFO:logisticloss:wbar: loss: 0.260544 errors: 580 (2.865 percent)
INFO:pegasos:Epoch 6 finished
INFO:logisticloss:w: loss: 0.262480 errors: 618 (3.053 percent)
INFO:logisticloss:wbar: loss: 0.260093 errors: 581 (2.870 percent)
INFO:pegasos:Epoch 7 finished
INFO:logisticloss:w: loss: 0.259453 errors: 574 (2.836 percent)
INFO:logisticloss:wbar: loss: 0.259790 errors: 577 (2.851 percent)
INFO:pegasos:Epoch 8 finished
INFO:logisticloss:w: loss: 0.259349 errors: 579 (2.860 percent)
INFO:logisticloss:wbar: loss: 0.259575 errors: 580 (2.865 percent)
INFO:pegasos:Epoch 9 finished
INFO:logisticloss:w: loss: 0.259183 errors: 572 (2.826 percent)
INFO:logisticloss:wbar: loss: 0.259416 errors: 580 (2.865 percent)
INFO:pegasos:Epoch 10 finished
INFO:logisticloss:w: loss: 0.260049 errors: 586 (2.895 percent)
INFO:logisticloss:wbar: loss: 0.259297 errors: 581 (2.870 percent)
INFO:pegasos:Epoch 11 finished
INFO:logisticloss:w: loss: 0.259039 errors: 579 (2.860 percent)
INFO:logisticloss:wbar: loss: 0.259200 errors: 581 (2.870 percent)
INFO:pegasos:Epoch 12 finished
INFO:logisticloss:w: loss: 0.259456 errors: 576 (2.846 percent)
INFO:logisticloss:wbar: loss: 0.259118 errors: 577 (2.851 percent)

### Correct regular saga

INFO:saga:Saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:saga:Epoch 0 finished
INFO:logisticloss: loss: 0.594437 errors: 6387 (31.553 percent)
INFO:saga:Epoch 1 finished
INFO:logisticloss: loss: 0.272780 errors: 754 (3.725 percent)
INFO:saga:Epoch 2 finished
INFO:logisticloss: loss: 0.260329 errors: 587 (2.900 percent)
INFO:saga:Epoch 3 finished
INFO:logisticloss: loss: 0.263351 errors: 635 (3.137 percent)
INFO:saga:Epoch 4 finished
INFO:logisticloss: loss: 0.259167 errors: 583 (2.880 percent)
INFO:saga:Epoch 5 finished
INFO:logisticloss: loss: 0.258398 errors: 572 (2.826 percent)
INFO:saga:Epoch 6 finished
INFO:logisticloss: loss: 0.258328 errors: 568 (2.806 percent)
INFO:saga:Epoch 7 finished
INFO:logisticloss: loss: 0.258302 errors: 568 (2.806 percent)
INFO:saga:Epoch 8 finished
INFO:logisticloss: loss: 0.258288 errors: 572 (2.826 percent)
INFO:saga:Epoch 9 finished
INFO:logisticloss: loss: 0.258278 errors: 570 (2.816 percent)
INFO:saga:Epoch 10 finished
INFO:logisticloss: loss: 0.258276 errors: 569 (2.811 percent)
INFO:saga:Epoch 11 finished
INFO:logisticloss: loss: 0.258276 errors: 569 (2.811 percent)
INFO:saga:Epoch 12 finished
INFO:logisticloss: loss: 0.258276 errors: 569 (2.811 percent)
INFO:saga:Epoch 13 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:saga:Epoch 14 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:saga:Epoch 15 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:saga:Epoch 16 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:saga:Epoch 17 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)

### Maybe working point saga:

INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.594177 errors: 6111 (30.190 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.267106 errors: 661 (3.265 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.260219 errors: 585 (2.890 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.263333 errors: 632 (3.122 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.259283 errors: 577 (2.851 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.258381 errors: 568 (2.806 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.258319 errors: 567 (2.801 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.258298 errors: 570 (2.816 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.258284 errors: 570 (2.816 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.258277 errors: 570 (2.816 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.258276 errors: 569 (2.811 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.258275 errors: 569 (2.811 percent)


### Less reg comparison

Step size 0.5
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:saga:Epoch 0 finished
INFO:logisticloss: loss: 0.672120 errors: 6765 (33.421 percent)
INFO:saga:Epoch 1 finished
INFO:logisticloss: loss: 0.108252 errors: 547 (2.702 percent)
INFO:saga:Epoch 2 finished
INFO:logisticloss: loss: 0.085840 errors: 402 (1.986 percent)
INFO:saga:Epoch 3 finished
INFO:logisticloss: loss: 0.081621 errors: 399 (1.971 percent)
INFO:saga:Epoch 4 finished
INFO:logisticloss: loss: 0.068291 errors: 295 (1.457 percent)
INFO:saga:Epoch 5 finished
INFO:logisticloss: loss: 0.061863 errors: 232 (1.146 percent)
INFO:saga:Epoch 6 finished
INFO:logisticloss: loss: 0.057865 errors: 205 (1.013 percent)
INFO:saga:Epoch 7 finished
INFO:logisticloss: loss: 0.054498 errors: 182 (0.899 percent)
INFO:saga:Epoch 8 finished
INFO:logisticloss: loss: 0.051856 errors: 165 (0.815 percent)
INFO:saga:Epoch 9 finished
INFO:logisticloss: loss: 0.049730 errors: 149 (0.736 percent)
INFO:saga:Epoch 10 finished
INFO:logisticloss: loss: 0.047836 errors: 133 (0.657 percent)
INFO:saga:Epoch 11 finished
INFO:logisticloss: loss: 0.046266 errors: 119 (0.588 percent)
INFO:saga:Epoch 12 finished
INFO:logisticloss: loss: 0.044931 errors: 104 (0.514 percent)
INFO:saga:Epoch 13 finished
INFO:logisticloss: loss: 0.043768 errors: 91 (0.450 percent)
INFO:saga:Epoch 14 finished
INFO:logisticloss: loss: 0.042770 errors: 87 (0.430 percent)
INFO:saga:Epoch 15 finished
INFO:logisticloss: loss: 0.041899 errors: 80 (0.395 percent)
INFO:saga:Epoch 16 finished
INFO:logisticloss: loss: 0.041127 errors: 73 (0.361 percent)
INFO:saga:Epoch 17 finished
INFO:logisticloss: loss: 0.040453 errors: 62 (0.306 percent)
INFO:saga:Epoch 18 finished
INFO:logisticloss: loss: 0.039846 errors: 61 (0.301 percent)
INFO:saga:Epoch 19 finished
INFO:logisticloss: loss: 0.039311 errors: 58 (0.287 percent)
INFO:saga:Point-saga finished

Step size 2
INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.658887 errors: 6492 (32.072 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.103172 errors: 542 (2.678 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.084037 errors: 410 (2.025 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.081444 errors: 408 (2.016 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.067575 errors: 294 (1.452 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.061121 errors: 228 (1.126 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.057257 errors: 198 (0.978 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.053973 errors: 178 (0.879 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.051409 errors: 163 (0.805 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.049353 errors: 145 (0.716 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.047507 errors: 131 (0.647 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.045977 errors: 116 (0.573 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.044676 errors: 100 (0.494 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.043543 errors: 90 (0.445 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.042570 errors: 85 (0.420 percent)
INFO:pointsaga:Epoch 15 finished
INFO:logisticloss: loss: 0.041722 errors: 78 (0.385 percent)
INFO:pointsaga:Epoch 16 finished
INFO:logisticloss: loss: 0.040968 errors: 71 (0.351 percent)
INFO:pointsaga:Epoch 17 finished
INFO:logisticloss: loss: 0.040309 errors: 62 (0.306 percent)
INFO:pointsaga:Epoch 18 finished
INFO:logisticloss: loss: 0.039717 errors: 59 (0.291 percent)
INFO:pointsaga:Epoch 19 finished
INFO:logisticloss: loss: 0.039196 errors: 57 (0.282 percent)
INFO:pointsaga:Point-saga finished

Lets try much larger step sizes:

INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 1.870535 errors: 7238 (35.757 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.125410 errors: 735 (3.631 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.066272 errors: 257 (1.270 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.084019 errors: 433 (2.139 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.072660 errors: 338 (1.670 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.046639 errors: 74 (0.366 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.045316 errors: 62 (0.306 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.042261 errors: 40 (0.198 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.040860 errors: 28 (0.138 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.039811 errors: 29 (0.143 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.038949 errors: 30 (0.148 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.038227 errors: 28 (0.138 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.037642 errors: 26 (0.128 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.037161 errors: 25 (0.124 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.036750 errors: 27 (0.133 percent)
INFO:pointsaga:Epoch 15 finished
INFO:logisticloss: loss: 0.036393 errors: 26 (0.128 percent)
INFO:pointsaga:Epoch 16 finished
INFO:logisticloss: loss: 0.036086 errors: 26 (0.128 percent)
INFO:pointsaga:Epoch 17 finished
INFO:logisticloss: loss: 0.035821 errors: 26 (0.128 percent)
INFO:pointsaga:Epoch 18 finished
INFO:logisticloss: loss: 0.035584 errors: 24 (0.119 percent)
INFO:pointsaga:Epoch 19 finished
INFO:logisticloss: loss: 0.035379 errors: 24 (0.119 percent)

step size 8:

INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 6.014381 errors: 7129 (35.219 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.696635 errors: 1611 (7.959 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.330612 errors: 503 (2.485 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.354378 errors: 973 (4.807 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.279713 errors: 908 (4.486 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.133650 errors: 124 (0.613 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.104868 errors: 85 (0.420 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.082750 errors: 52 (0.257 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.063702 errors: 30 (0.148 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.052968 errors: 28 (0.138 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.046169 errors: 30 (0.148 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.041066 errors: 24 (0.119 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.038250 errors: 23 (0.114 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.036670 errors: 23 (0.114 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.035595 errors: 25 (0.124 percent)
INFO:pointsaga:Epoch 15 finished
INFO:logisticloss: loss: 0.034952 errors: 22 (0.109 percent)
INFO:pointsaga:Epoch 16 finished
INFO:logisticloss: loss: 0.034552 errors: 23 (0.114 percent)
INFO:pointsaga:Epoch 17 finished
INFO:logisticloss: loss: 0.034276 errors: 20 (0.099 percent)
INFO:pointsaga:Epoch 18 finished
INFO:logisticloss: loss: 0.034095 errors: 24 (0.119 percent)
INFO:pointsaga:Epoch 19 finished
INFO:logisticloss: loss: 0.033981 errors: 25 (0.124 percent)

Using step size 16 seems to be destabalizing it a but:

INFO:pointsaga:Gamma: 1.60e+01, prox_conversion_factor: 0.99998400, 1-reg*gamma: 0.99998400
INFO:pointsaga:Point-saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 10.880857 errors: 7035 (34.754 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 1.783385 errors: 2217 (10.952 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.744621 errors: 653 (3.226 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.924417 errors: 2037 (10.063 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.416542 errors: 1017 (5.024 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.183182 errors: 248 (1.225 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.145502 errors: 411 (2.030 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.084493 errors: 144 (0.711 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.053535 errors: 43 (0.212 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.043074 errors: 35 (0.173 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.039048 errors: 41 (0.203 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.036123 errors: 37 (0.183 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.035027 errors: 30 (0.148 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.034429 errors: 27 (0.133 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.034100 errors: 27 (0.133 percent)
INFO:pointsaga:Epoch 15 finished
INFO:logisticloss: loss: 0.033958 errors: 27 (0.133 percent)
INFO:pointsaga:Epoch 16 finished
INFO:logisticloss: loss: 0.033878 errors: 20 (0.099 percent)
INFO:pointsaga:Epoch 17 finished
INFO:logisticloss: loss: 0.033810 errors: 22 (0.109 percent)
INFO:pointsaga:Epoch 18 finished
INFO:logisticloss: loss: 0.033780 errors: 24 (0.119 percent)
INFO:pointsaga:Epoch 19 finished
INFO:logisticloss: loss: 0.033769 errors: 23 (0.114 percent)
INFO:pointsaga:Point-saga finished

Compare to 0.033981, still lower at the end though. Good.

Regular saga seems to be able to handle a little larger step size:

INFO:saga:Saga starting, npoints=20242, ndims=47237
INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:saga:Epoch 0 finished
INFO:logisticloss: loss: 1.707874 errors: 7237 (35.752 percent)
INFO:saga:Epoch 1 finished
INFO:logisticloss: loss: 0.174765 errors: 1148 (5.671 percent)
INFO:saga:Epoch 2 finished
INFO:logisticloss: loss: 0.071428 errors: 312 (1.541 percent)
INFO:saga:Epoch 3 finished
INFO:logisticloss: loss: 0.083095 errors: 397 (1.961 percent)
INFO:saga:Epoch 4 finished
INFO:logisticloss: loss: 0.065822 errors: 266 (1.314 percent)
INFO:saga:Epoch 5 finished
INFO:logisticloss: loss: 0.050108 errors: 112 (0.553 percent)
INFO:saga:Epoch 6 finished
INFO:logisticloss: loss: 0.046697 errors: 75 (0.371 percent)
INFO:saga:Epoch 7 finished
INFO:logisticloss: loss: 0.053051 errors: 123 (0.608 percent)
INFO:saga:Epoch 8 finished
INFO:logisticloss: loss: 0.042715 errors: 46 (0.227 percent)
INFO:saga:Epoch 9 finished
INFO:logisticloss: loss: 0.041346 errors: 40 (0.198 percent)
INFO:saga:Epoch 10 finished
INFO:logisticloss: loss: 0.040668 errors: 34 (0.168 percent)
INFO:saga:Epoch 11 finished
INFO:logisticloss: loss: 0.039432 errors: 37 (0.183 percent)
INFO:saga:Epoch 12 finished
INFO:logisticloss: loss: 0.038841 errors: 32 (0.158 percent)
INFO:saga:Epoch 13 finished
INFO:logisticloss: loss: 0.037924 errors: 25 (0.124 percent)
INFO:saga:Epoch 14 finished
INFO:logisticloss: loss: 0.037404 errors: 26 (0.128 percent)
INFO:saga:Epoch 15 finished
INFO:logisticloss: loss: 0.036974 errors: 25 (0.124 percent)
INFO:saga:Epoch 16 finished
INFO:logisticloss: loss: 0.036594 errors: 25 (0.124 percent)
INFO:saga:Epoch 17 finished
INFO:logisticloss: loss: 0.036271 errors: 24 (0.119 percent)
INFO:saga:Epoch 18 finished
INFO:logisticloss: loss: 0.035985 errors: 23 (0.114 percent)
INFO:saga:Epoch 19 finished
INFO:logisticloss: loss: 0.035737 errors: 23 (0.114 percent)

##### Verification of final

Runing with with lag turned on:

INFO:logisticloss: loss: 0.693147 errors: 20242 (100.000 percent)
INFO:pointsaga:Epoch 0 finished
INFO:logisticloss: loss: 0.654565 errors: 6473 (31.978 percent)
INFO:pointsaga:Epoch 1 finished
INFO:logisticloss: loss: 0.133410 errors: 546 (2.697 percent)
INFO:pointsaga:Epoch 2 finished
INFO:logisticloss: loss: 0.121083 errors: 419 (2.070 percent)
INFO:pointsaga:Epoch 3 finished
INFO:logisticloss: loss: 0.122882 errors: 411 (2.030 percent)
INFO:pointsaga:Epoch 4 finished
INFO:logisticloss: loss: 0.113839 errors: 315 (1.556 percent)
INFO:pointsaga:Epoch 5 finished
INFO:logisticloss: loss: 0.110785 errors: 267 (1.319 percent)
INFO:pointsaga:Epoch 6 finished
INFO:logisticloss: loss: 0.109721 errors: 249 (1.230 percent)
INFO:pointsaga:Epoch 7 finished
INFO:logisticloss: loss: 0.108878 errors: 236 (1.166 percent)
INFO:pointsaga:Epoch 8 finished
INFO:logisticloss: loss: 0.108344 errors: 223 (1.102 percent)
INFO:pointsaga:Epoch 9 finished
INFO:logisticloss: loss: 0.107985 errors: 217 (1.072 percent)
INFO:pointsaga:Epoch 10 finished
INFO:logisticloss: loss: 0.107701 errors: 212 (1.047 percent)
INFO:pointsaga:Epoch 11 finished
INFO:logisticloss: loss: 0.107514 errors: 202 (0.998 percent)
INFO:pointsaga:Epoch 12 finished
INFO:logisticloss: loss: 0.107379 errors: 199 (0.983 percent)
INFO:pointsaga:Epoch 13 finished
INFO:logisticloss: loss: 0.107281 errors: 192 (0.949 percent)
INFO:pointsaga:Epoch 14 finished
INFO:logisticloss: loss: 0.107209 errors: 187 (0.924 percent)
INFO:pointsaga:Epoch 15 finished
INFO:logisticloss: loss: 0.107155 errors: 183 (0.904 percent)
INFO:pointsaga:Epoch 16 finished
INFO:logisticloss: loss: 0.107115 errors: 181 (0.894 percent)
INFO:pointsaga:Epoch 17 finished
INFO:logisticloss: loss: 0.107085 errors: 179 (0.884 percent)
INFO:pointsaga:Epoch 18 finished
INFO:logisticloss: loss: 0.107062 errors: 180 (0.889 percent)
INFO:pointsaga:Epoch 19 finished
INFO:logisticloss: loss: 0.107045 errors: 177 (0.874 percent)
INFO:pointsaga:Point-saga finished

Ok, non-lag version appears to give identical results, so probably ok.



