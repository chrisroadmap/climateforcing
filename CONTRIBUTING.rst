Contributing to climateforcing
------------------------------

Thanks for taking the time to contribute!

Bug reports
===========
- Please raise an issue, giving as much detail as possible about your bug, and a minimal working example that reproduces the bug.

Feature requests
================
- Feel free to raise an issue if there's something you'd like to see, but things might get done quicker if you try and code it up yourself and make a pull request.

Contributing
============
The best way to contribute is to fork the repository, make a new branch, then pull request. 
Please write a test that covers your new code (add to ``tests``). After implementing your feature the overall code coverage should not decrease (see below).
Please also add a line to the ``CHANGELOG.rst`` file that describes what your pull request does and what it adds/changes/fixes.

Code is automatically formatted and linted for style. Before committing a branch, you can run all of the tests locally:

.. code-block::

    $ make checks

This will fail if some of the checks don't pass, and it will also run the test suite and produce an overall coverage report. The majority of the formatting can be automated with

.. code-block::

    $ make black
    $ make isort
    
You will have to satisfy ``pylint`` and ``flake8`` yourself, but hopefully this is not too painful. (You get used to it).

Please also check that the master branch is still up to date before you commit your branch, and rebase if necessary. I find `this guide <https://medium.com/@topspinj/how-to-git-rebase-into-a-forked-repo-c9f05e821c8a>`_ very useful.
