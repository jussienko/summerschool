#include <math.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <mpi.h>

#include "heat.h"

int rows=200;
int cols=200;
double a = 0.5;
parallel_data parallel;
field current;
field previous;

void testgenerate(void) {
    double result;
    parallel_setup(&parallel, rows, cols);
    set_field_dimensions(&current, rows, cols, &parallel);
    generate_field(&current, &parallel);
    MPI_Allreduce(&current.data[2][2], &result, 1, MPI_DOUBLE, 
               MPI_SUM, MPI_COMM_WORLD);

    CU_ASSERT_DOUBLE_EQUAL(result, 260.0, 0.00001);
}

void testevolve(void) {
    double dx2;
    double dy2;
    double dt; 
    double result;

    set_field_dimensions(&previous, rows, cols, &parallel);
    generate_field(&current, &parallel);
    allocate_field(&previous);
    copy_field(&current, &previous);

    dx2 = current.dx *  current.dy;
    dy2 = current.dy *  current.dy;
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
    evolve(&current, &previous, a, dt);
    MPI_Allreduce(&current.data[2][2], &result, 1, MPI_DOUBLE, 
               MPI_SUM, MPI_COMM_WORLD);
    CU_ASSERT_DOUBLE_EQUAL(result, 260.0, 0.00001);
}

int init_suite(void)
{
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    if (ntasks != 4) {
       printf("Test should be run with 4 mpi tasks\n");
       return -1;
    }
    else
    {
       return 0;
    }
}

int main(int argc, char **argv)
{
   int rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   // Redirect stdout and stderr if not master
   if (rank != 0) {
       freopen( "/dev/null", "w", stdout);
       freopen( "/dev/null", "w", stderr);
   }

   CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Heat equation", init_suite, NULL);
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of generate", testgenerate)) ||
       (NULL == CU_add_test(pSuite, "test of evolve", testevolve)))
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   MPI_Finalize();
   return CU_get_error();
}
