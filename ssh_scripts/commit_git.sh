#!/bin/bash

#!/bin/bash 
#######################################################################################
#Script Name    : commit_git_mod.sh
#Description    : regularly commits changes from given repo 
#Args           : None      
#Author         : Arianna Taormina
#Email          : ariannataormina89@gmail.com
#License        : None	
#######################################################################################

MESSAGE="Update_$(date)"
DESTINATION_BRANCH="main"
LOCAL_REPO="/Users/ariannataormina/Documents/GitHub/multiagent_system_"
GIT_REPO="git@github.com:arita89/multiagent_system.git"
SEPARATOR="#######################################"
cd "$LOCAL_REPO"


# authenticate onto github
# gh auth login

# pull changes before committing them
git pull

echo " "
echo "$SEPARATOR"

# evaluate return status of last command
if [ $? -ne 0 ];
then
    echo "You are not in the proper directory"
else
    echo "You're about to push your changes"
    echo "from:$LOCAL_REPO"
    echo "to:$GIT_REPO"
    echo " "
    git remote set-url origin "$GIT_REPO"
    git add .     
    #read -p "What's the message you desire to type? " MESSAGE
    git commit -a -m "$MESSAGE"
    #read -p "What's the name of the destination branch? " DESTINATION_BRANCH
    git push -u origin $DESTINATION_BRANCH
    echo " "
    echo "Hope to see you again!"
    echo "$MESSAGE"
    echo " "
    echo "$SEPARATOR"
fi