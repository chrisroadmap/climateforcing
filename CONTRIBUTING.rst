Contributing to climateforcing
------------------------------

Thanks for taking the time to contribute!

Bug reports
-----------
- Please raise an issue, giving as much detail as possible about your bug, and a minimal working example that reproduces the bug.

Feature requests
----------------
- Feel free to raise an issue if there's something you'd like to see, but things might get done quicker if you try and code it up yourself and make a pull request.

Contributing
------------
The best way to contribute is to fork the repository, make a new branch, then pull request. 
Please write a test that covers your new code (add to ``tests``). After implementing your feature the overall code coverage should not decrease (see below).
Please also add a line to the ``CHANGELOG.rst`` file that describes what your pull request does and what it adds/changes/fixes.

Please also check that the master branch is still up to date before you commit your branch, and rebase if necessary. I find `this guide <https://medium.com/@topspinj/how-to-git-rebase-into-a-forked-repo-c9f05e821c8a>`_ very useful.

Code style
==========
Code is automatically formatted and linted for style. Before committing a branch, you can run all of the pre-commit checks locally:

.. code-block::

    $ make checks

This will fail if some of the checks don't pass, and it will also run the test suite and produce an overall coverage report. The majority of the formatting can be automated with

.. code-block::

    $ make black
    $ make isort
    
You will have to satisfy ``pylint``, ``pydocstyle`` and ``flake8`` yourself, but hopefully this is not too painful. (You get used to it).

Tests and coverage
==================
Please write a test that verifies your code does what you intend - this will help ensure that when new features get added, nothing is inadvertently broken. It is particuarly helpful for numerical functions to include small examples and to check that the intended result is obtained. See the ``tests`` directory of the repository for existing functions.

You can run the tests separately to the rest of the checks:

.. code-block::

    $ make test
    
This will produce the results of any failing tests and also a coverage report which shows which lines of the code are not being picked up by tests. It is best to aim for 100% coverage for all new modules. The tests will fail if 90% total coverage is not reached (this is likely to be increased in future).

Documentation
=============
Please document your functions, classes and modules using the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html/>`_ convention. Again, a brief scout around the existing repository should put you on the right track.
