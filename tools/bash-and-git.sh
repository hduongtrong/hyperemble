## This file isn't meant to be run all at once. It just contains various useful git/bash configurations.

## Paste the following line into your terminal to add a prettier version of `git log` called `git lg`
## This is a persistent configuration command - you only need to run it once.
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

## Copy the next section into ~/.bashrc to show branch in bash prompt.
function parse_git_branch () {                                                
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'          
}                                                                             
                                                                              
RED="\[\033[0;31m\]"                                                          
YELLOW="\[\033[0;33m\]"                                                       
GREEN="\[\033[0;32m\]"                                                        
NO_COLOR="\[\033[0m\]"                                                        
PS1="$GREEN\u@\h$NO_COLOR:\w$YELLOW\$(parse_git_branch)$NO_COLOR\$ "          
