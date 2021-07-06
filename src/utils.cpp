#include "headers/utils.h"

DLLEXPORT void init_random(){
    srand(time(nullptr));
}

// how to use it :
//  int * len = (int*)malloc(sizeof(int));
//  char ** rslt = split("test;pour;voir si;tout;fonctionne corrextement",len);
//  for(int i =0; i<*len;i++){
//      printf(" %s\n",rslt[i]);
//  }
char ** split(char* str, int* array_len, char delimiter){
    int i, j, ctr, maxLenSubstring=0, actuelLenSubstring=0, cpt=1;

    // count delimiter number
    for(i=0;i<=(strlen(str));i++){
        actuelLenSubstring++;
        if(str[i] == delimiter){
            cpt++;
            if(actuelLenSubstring > maxLenSubstring){
                maxLenSubstring = actuelLenSubstring;
            }
            actuelLenSubstring = 0;
        }
    }
    char ** stringArray = (char**)malloc(sizeof(char*) * cpt);
    int subLength = maxLenSubstring + 1;
    char* substring = (char*)malloc(sizeof(char)*subLength);

    * array_len = cpt; //return length of the real return

    j=0; ctr=0;
    for(i=0;i<=(strlen(str));i++)
    {
        if(str[i]==delimiter||str[i]=='\0')
        {
            substring[j]='\0';
            stringArray[ctr] = (char *)malloc(sizeof(char)*(j+1));
            strcpy(stringArray[ctr], substring);

            ctr++;  //for next word
            j=0;    //for next word, init index to 0
            memset(substring, ' ', strlen(substring)); // reset var substring
        }
        else
        {
            substring[j]=str[i];
            j++;
        }
    }
    return stringArray;
}


