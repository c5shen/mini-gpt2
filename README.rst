Mini GPT-2
==========

:Developer:
    Chengze Shen

Overview
--------
The code is adapted from Andrej Karpathy's Youtube tutorial on how to implement
and train a (small) GPT-2 model with a decoder-only transformer architecture.
The tutorial can be found at
`this Youtube link <https://www.youtube.com/watch?v=kCc8FmEb1nY>`__.

Files
-----
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - filename
     - description
   * - ``train_model.py``
     - Implementation of the GPT-2 model used by ``train.py``. 
   * - ``train.py``
     - (training) The original implementation from Andrej's video.
       Takes a directory name as the first argument to read the training data
       and output the trained model.
       E.g., ``python train.py tiny_shakespeare`` will find a file named
       ``input.txt`` under the given folder as input. Trained model is written
       to the same directory as ``train.model``. If an existing trained model
       is present in the directory, will not overwrite but will save the new
       trained model as ``train.model.X``, ``X=1,2,...``.
   * - ``train_eval.py``
     - (evaluation) For evaluating a trained model (``train.model``).
       Takes a directory name as the first argument to read the trained model.
   * - ``gpt2.py``
     - (training) My implementation with customizations of the GPT-2 model.
       With both the model implementation and the code for training.
       Takes a directory name as the first argument for reading training data
       and outputting trained model.
   * - ``gpt2_eval.py``
     - (evaluation) For evaluating a trained model (``gpt2.model``).
       Takes a directory name as the first argument to read the trained model.
   * - ``gpt2_test.py``
     - (evaluation) More advanced and interactive code for evaluating a
       trained model (``gpt2.model``). Allows continuous reading input from
       the user and saves context from previous user inputs.

Customization
-------------
I did some customizations based on the original code presented by Andrej. See
``gpt2.py`` and ``gpt2_test.py`` for more details.

Usage
-----

Training
++++++++
Training with ``gpt2.py`` on ``all_shakespeare``:

.. code:: bash

   python3 gpt2.py all_shakespeare/

Evaluation
++++++++++
Evaluating an existing trained model (``gpt2.model``) from ``all_shakespeare``.
Note that ``gpt2.model`` must present under the target directory. 

.. code:: bash

   python3 gpt2_test.py all_shakespeare/
