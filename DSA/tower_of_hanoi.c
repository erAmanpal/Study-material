/* C program for Tower of Hanoi using Recursion */
 
#include <stdio.h>
void towers(int, char, char, char);

int main()
{
    int num;
    printf("Enter the number of disks : ");
    scanf("%d", &num);
    printf("The sequence of moves involved in the Tower of Hanoi are :\n");
    towers(num, 'A', 'C', 'B');
    return 0; 
}
 
void towers(int num, char BEG, char END, char AUX)
{
    if (num == 1)
    {
        printf("\n Move disk 1 from peg %c to peg %c", BEG, END);
        return;
    }
    towers(num - 1, BEG, AUX, END);
    printf("\n Move disk %d from peg %c to peg %c", num, BEG, END);
    towers(num - 1, AUX, END, BEG); 
}
