#--------------set path for gcc
set path=(/usr/bin $path)
# -----------  MPICH configuration
setenv MPICH_HOME /usr/local/mpich2
set path=($MPICH_HOME/bin $path)
set path=($MPICH_HOME/sbin $path)
setenv TMPDIR /tmp

