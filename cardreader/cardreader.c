#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to print the card details
void printCardDetails(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    printf("Card details:\n");
    char line[200];
    while (fgets(line, sizeof(line), file)) {
        printf("%s", line);
    }

    fclose(file);
}

int main() {
    char choice;
    char name[100], language[100], condition[100], edition[100];
    float price;

    while (1) {
        printf("\nOptions:\n");
        printf("1. Add card\n");
        printf("2. Print cards\n");
        printf("3. Exit\n");
        printf("Enter your choice: ");
        scanf(" %c", &choice);

        switch (choice) {
            case '1':
                printf("\nEnter the name of the Yu-Gi-Oh card: ");
                getchar(); // Clear input buffer
                fgets(name, sizeof(name), stdin);
                printf("Enter the language of the card: ");
                fgets(language, sizeof(language), stdin);
                printf("Enter the condition of the card: ");
                fgets(condition, sizeof(condition), stdin);
                printf("Enter the edition of the card: ");
                fgets(edition, sizeof(edition), stdin);
                printf("Enter the price of the card: ");
                scanf("%f", &price);

                // Open the file in append mode
                FILE *file = fopen("yugioh_cards.txt", "a");
                if (file == NULL) {
                    printf("Error opening file!\n");
                    exit(1);
                }

                // Write data to the file
                fprintf(file, "Name: \"%s\" (Language: \"%s\")\n", name, language);
                fprintf(file, "Condition: \"%s\"", condition);
                fprintf(file, "Edition: \"%s\"", edition);
                fprintf(file, "Price: %.2f\n", price);

                // Close the file
                fclose(file);

                printf("Card added successfully!\n");
                break;

            case '2':
                printCardDetails("yugioh_cards.txt");
                break;

            case '3':
                printf("Exiting program...\n");
                exit(0);

            default:
                printf("Invalid choice! Please enter again.\n");
                break;
        }
    }

    return 0;
}

