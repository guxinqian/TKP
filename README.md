## [Temporal Knowledge Propagation for Image-to-Video Person Re-identification](https://arxiv.org/abs/1908.03885)

#### Requirements: Python=3.6 and Pytorch=1.0.0



### Training and test

  ```Shell
  # For MARS
  python train.py --root /data/datasets/ -d mars --save_dir log-mars
  python test.py --root /data/datasets/ -d mars --resume log-mars/best_model.pth.tar --save_dir log-mars
  
  # For DukeMTMC-VideoReID
  python train.py --root /data/datasets/ -d dukevid --save_dir log-duke
  python test.py --root /data/datasets/ -d dukevid --resume log-duke/best_model.pth.tar --save_dir log-duke
  
  # For iLIDS-VID (If you use the pretrained model on Duke, you will get a much higher results than that reported in our paper.)
  python main_ilids.py --root /data/datasets/ --save_dir log-ilids
  ```




### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @inproceedings{gu2019TKP,
      title={Temporal Knowledge Propagation for Image-to-Video Person Re-identification},
      author={Gu, Xinqian and Ma, Bingpeng and Chang, Hong and Shan, Shiguang and Chen, Xilin},
      booktitle={ICCV},
      year={2019},
    }
