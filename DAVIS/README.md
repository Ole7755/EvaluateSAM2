DAVIS: A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS)
============================================================================================

Package containing input/output data used in: *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation* [DAVIS](https://graphics.ethz.ch/~perazzif/davis/index.html).

Documentation
-----------------

The directory is structured as follows:

 * `ROOT/JPEGImages`: Set of input video sequences provided in the form of JPEG images.
   Video sequences are available at 1080p and 480p resolution.

 * `ROOT/Annotations`: Set of manually annotated binary images providing reference segmentation
   for the foreground objects. Annotations are available at 1080p and 480p resolution.

 * `ROOT/db_info.yml`: File containing information about video sequences.

Credits
---------------
All sequences if not stated differently are owned by the authors of `DAVIS` and are
licensed under Creative Commons Attributions 4.0 License, see [Terms of Use].

The following sequences were downloaded from YouTube:
  * dog-agility     [www.husse.it](https://www.youtube.com/watch?v=LEz1VzUKTQk)
  * breakdance      [Игорь Калмыков](https://www.youtube.com/watch?v=5Ys8Gv3uPGA)
  * drift-chicane   [TOYO TIRES JAPAN](https://www.youtube.com/watch?v=w8jv-WSgKzE)
  * drift-turn      [TOYO TIRES JAPAN](https://www.youtube.com/watch?v=w8jv-WSgKzE)
  * drift-straight  [Kreon Films](https://www.youtube.com/watch?v=oBXdW2g25Vg)
  * motocross-bumps [Michał Jaroszczyk](https://www.youtube.com/watch?v=lWYG_xLG6YU)
  * motocross-jump  [Michał Jaroszczyk](https://www.youtube.com/watch?v=lWYG_xLG6YU)
  * parkour         [Juho Kuusisaari](https://www.youtube.com/watch?v=cpcUARb6p0c)

Please refer to the provided links for their terms-of-use.

Citation
--------------

Please cite `DAVIS` in your publications if it helps your research:

    `@inproceedings{Perazzi_CVPR_2016,
      author    = {Federico Perazzi and
                   Jordi Pont-Tuset and
                   Brian McWilliams and
                   Luc Van Gool and
                   Markus Gross and
                   Alexander Sorkine-Hornung},
      title     = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2016}
    }`

Terms of Use
--------------

`DAVIS` is released under the Creative Commons License:
  [CC BY-NC](http://creativecommons.org/licenses/by-nc/4.0).

In synthesis, users of the data are free to:

1. **Share** - copy and redistribute the material in any medium or format.
2. **Adapt** - remix, transform, and build upon the material.

The licensor cannot revoke these freedoms as long as you follow the license terms.

Contacts
------------------
- [Federico Perazzi](https://graphics.ethz.ch/~perazzif)
- [Jordi Pont-Tuset](http://jponttuset.github.io)

