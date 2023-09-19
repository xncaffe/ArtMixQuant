# ArtMixQuant

## Introduction

This is an automatic hybrid quantization solution adapted to the Art.Studio tool chain.
Use it to search for a locally optimal 8bit and 16bit mixed quantization configuration scheme
This solution can take into account both accuracy and speed.
When you encounter insufficient accuracy of 8bit and 16bit cannot meet the speed requirements during deployment, you can try to use this tool.
Art.Studio can contact Kuxin Microelectronics.

## Usage

### Installation

1. Get the Art.Studio tool chain
2. ```
   git clone https://github.com/xncaffe/ArtMixQuant.git
   cd ArtMixQuant
   pip install -r requirements.txt
   ```

### Mode

1. #### fast mode


   ```
   python run_mix_quant.py -a /your/path/Art.Studio-v0.. -ini /your/path/your_net.ini -o ./mixOutput/your_net/ -pm 0 -w 10 --UseOpt
   ```

   ** The '-w' represents the number of execution threads, and the default is 1. Use multithreading to speed up searches

   After the operation is completed, you can get the corresponding precision configuration json file and a quantitative performance evaluation table "output_0.xlsx"
2. #### normal mode


   ```
   python run_mix_quant.py -a /your/path/Art.Studio-v0.. -ini /your/path/your_net.ini -o ./mixOutput/your_net/ -pm 1 -w 20 -r 10000 --UseOpt
   ```

   ** The '-w' is same with fast mode. The '-r' represents the number of iterations, similar to the number of training iterations of CNN.

   The normal mode can visualize the search results in real time. If you get the expected performance data, you can use ctrl+c to terminate the program in advance. Performance data is also stored in output_0.xlsx.

   To visualize the current status, please refer to the two png images in ./mixOutput/your_net/

## Note

* In normal mode, the top10 in the output_0.xlsx table are updated and displayed in real time.

* In normal mode, every 100 iterations is a tile, and reward loss statistics are printed. As shown in the figure below, it can be seen that the fluctuation of reward loss increases, proving that it gradually converges;
* Support statistics of peak physical memory usage.
* Support real-time update output_0.xlsx.
* When the number of threads is configured unreasonably, it supports automatic adjustment to a reasonable state.


For detailed usage plans, please contact Artosyn official.
