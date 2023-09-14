#include <stdio.h>
#include <math.h>
#include <stdlib.h>



typedef struct {
    double a, b, c;
} FuzzyNumber;



FuzzyNumber createFuzzyNumber(double num1, double num2, double num3) {
    FuzzyNumber fuzzy;
    fuzzy.a = num1;
    fuzzy.b = num2;
    fuzzy.c = num3;
    return fuzzy;
}

char* fuzzy2str(FuzzyNumber num) {
    char* str = (char*)malloc(50);
    sprintf(str, "(%.2lf,%.2lf,%.2lf)", num.a, num.b, num.c);
    return str;
}

FuzzyNumber fuzzy_prod(FuzzyNumber num1, FuzzyNumber num2) {
    FuzzyNumber num3;
    num3.a = num1.a * num2.a;
    num3.b = num1.b * num2.b;
    num3.c = num1.c * num2.c;
    return num3;
}

FuzzyNumber fuzzy_div(FuzzyNumber num1, FuzzyNumber num2) {
    FuzzyNumber num3;
    if (num2.c == 0) {
        num2.c = num2.c + 0.01;
    }
    if (num2.b == 0) {
        num2.b = num2.b + 0.01;
    }
    if (num2.a == 0) {
        num2.a = num2.a + 0.01;
    }
    num3.a = num1.a / num2.c;
    num3.b = num1.b / num2.b;
    num3.c = num1.c / num2.a;
    return num3;
}

double fuzzy_distance(FuzzyNumber num1, FuzzyNumber num2) {
    return sqrt(1 / 3 * ((num1.a - num2.a) * (num1.a - num2.a) + (num1.b - num2.b) * (num1.b - num2.b) + (num1.c - num2.c) * (num1.c - num2.c)));
}

void fuzzy_min_max( int n,  int m, int j, FuzzyNumber list1[][m], FuzzyNumber* rm1, FuzzyNumber* rm2) {
    double max1 = list1[0][j].c;
    double min1 = list1[0][j].a;
    for (int i = 0; i < n; i++) {
        if (list1[i][j].c > max1) {
            max1 = list1[i][j].c;
        }
        if (list1[i][j].a < min1) {
            min1 = list1[i][j].a;
        }
    }
    *rm1 = createFuzzyNumber(max1, max1, max1);
    *rm2 = createFuzzyNumber(min1, min1, min1);
}

void fuzzy_norm_fun(int n, int m, FuzzyNumber all_ratings[n][m], int impact[],  FuzzyNumber norm_num[n][m]) {
    FuzzyNumber po[m], no[m];
    for (int j = 0; j < m; j++) {
        fuzzy_min_max(n, m, j, all_ratings, &po[j], &no[j]);
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (impact[j] == 1) {
                norm_num[i][j] = fuzzy_div(all_ratings[i][j], po[j]);
            } else {
                norm_num[i][j] = fuzzy_div(no[j], all_ratings[i][j]);
            }
        }
    }
}

void func_dist_fnis_fpis(  int n, int m, FuzzyNumber matr[][m], FuzzyNumber fpis[],  FuzzyNumber fnis[], double dist_pis[], double dist_nis[]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            dist_pis[i] += fuzzy_distance(fpis[j], matr[i][j]);
            dist_nis[i] += fuzzy_distance(fnis[j], matr[i][j]);
        }
    }
}


void fuzzy_w_norm_fun( int n, int m, FuzzyNumber all_ratings[][m], FuzzyNumber weight[], FuzzyNumber test[n][m]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            test[i][j] = fuzzy_prod(all_ratings[i][j], weight[j]);
        }
    }
}

double roundToThreeDecimalPlaces(double num) {
    return round(num * 1000) / 1000;
}

void f_topsis( int n, int m, FuzzyNumber all_ratings[][m], int impact[], double CC[]) {
    
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < m; j++) {
    //        printf("%s\t", fuzzy2str(all_ratings[i][j]));
    //    }
    //}
    //printf("n=%d,m=%d\n", n, m);
    FuzzyNumber weight[] = {
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(3, 5, 7),
        createFuzzyNumber(9, 9, 9)
    };

    FuzzyNumber fuzzy_norm_matrix[n][m];
    fuzzy_norm_fun( n, m, all_ratings, impact, fuzzy_norm_matrix);

    FuzzyNumber fuzzy_w_norm_matrix[n][m];
    fuzzy_w_norm_fun( n, m, fuzzy_norm_matrix, weight, fuzzy_w_norm_matrix);

    FuzzyNumber fpis[m], fnis[m];
    for (int j = 0; j < n; j++) {
        fuzzy_min_max(n, m, j, fuzzy_w_norm_matrix, &fpis[j], &fnis[j]);
    }

    double a_plus[m], a_minus[m];
    func_dist_fnis_fpis( n, m, fuzzy_w_norm_matrix, fpis, fnis, a_plus, a_minus);

    double newcc[n];
    
    for (int i = 0; i < n; i++) {
        
        newcc[i] = roundToThreeDecimalPlaces(a_minus[i] / (a_plus[i] + a_minus[i] + 0.01));
        //printf("i= %d, a_minus[i] = %.2lf, a_plus[i] = %.2lf, a_minus[i]= %.2lf, newcc[i]= %.2lf \n", i, a_minus[i], a_plus[i], a_minus[i], newcc[i]);
        //printf("%.2lf\n", newcc[i]);
        CC[i] = newcc[i];
    }
}


