!Interface to libcloudph++: lagrangian scheme (super-droplet) 

module mphys_libcloud_lgr

  Use parameters, only : nz, dt, nx
  Use column_variables
  Use physconst, only : p0, r_on_cp, pi

  Use diagnostics, only: save_dg, i_dgtime
  use iso_c_binding, only: c_funptr, c_f_procpointer, c_null_char, c_double, c_float

  Implicit None
  
  interface
    function micro_step_py(i_dgtime, dt, size_z, size_x,  & 
                        th_ar, qv_ar, rhof_ar, rhoh_ar, &
                        vf_ar, vh_ar, wf_ar, wh_ar,     &
                        xf_ar, zf_ar, xh_ar, zh_ar, tend_th_ar, tend_qv_ar) bind(c)
      use iso_c_binding, only: c_double, c_int, c_float, c_bool
      Use parameters, only : nx, nz
      logical(c_bool) :: micro_step_py
      integer(c_int), intent(in), value :: i_dgtime, size_x, size_z
      real(c_float), intent(in), value :: dt
                                           
      real(c_double),  intent(inout):: th_ar(nz, 0:nx+1), qv_ar(nz, 0:nx+1),  &
                                       rhof_ar(nz), rhoh_ar(nz),              &
                                       vh_ar(nz, 0:nx+1), vf_ar(nz, 0:nx+1),  &
                                       wf_ar(nz, 0:nx+1), wh_ar(nz, 0:nx+1),  &
                                       xf_ar(0:nx+1), zf_ar(nz), xh_ar(0:nx+1), zh_ar(nz), &
                                       tend_th_ar(nz, 0:nx+1), tend_qv_ar(nz, 0:nx+1)

    end

    subroutine load_ptr(fname, ptr) bind(c)
      use iso_c_binding, only: c_funptr, c_char
      character(kind=c_char), dimension(*), intent(in) :: fname
      type(c_funptr), intent(out) :: ptr
    end
  end interface

  type(c_funptr) :: cptr
  procedure(micro_step_py), pointer :: fptr => NULL()

contains

  Subroutine mphys_libcloud_lgr_interface

    character(10) :: uid, pid

    ! do the below only once
    if (associated(fptr) .eqv. .false.) then 
      ! assert for numerical precision  
      if (wp.ne.c_double) stop("KiD does not use double precision!")          
      if (sizeof(dt).ne.c_float) stop("dt in KiD is not a float!")

      ! load pointer to Python micro_step() routine (getuid() and getpid() are GNU extensions)
      write (uid, "(I10.0)") getuid()
      write (pid, "(I10.0)") getpid()
      call load_ptr("/tmp/micro_step-" // trim(adjustl(uid)) // "-" // trim(adjustl(pid)) // ".ptr" // c_null_char,cptr)

      ! associate the C pointer with the F pointer
      call c_f_procpointer(cptr, fptr)
    end if

    print*, "qv(10,20) w fortranie", qv(10,20) 
    ! do the below every timestep
    if (.not. fptr(i_dgtime, dt, nz, nx+2 , &
                   theta, qv, rho, rho_half, & 
                   v, v_half, w, w_half, x, z, x_half, z_half, dTheta_mphys, dqv_mphys) &
    ) stop("Error in Python!!!")

  end Subroutine mphys_libcloud_lgr_interface

end module mphys_libcloud_lgr
