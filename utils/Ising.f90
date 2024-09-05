        program main
        implicit none
        integer, parameter :: N=6
        integer, parameter :: Nz=1
        real, parameter :: J=1.00,J2=0
        real, parameter :: J_p=0.0
        integer :: s(N,N,Nz),s_p(N,N,Nz),E,M,k1,k2,k3,I,ii,jj,kk,I2
        integer*8 :: A,ss(N,N,Nz),ss_p(N,N,Nz),Z
        integer :: anneal_steps, decorrelation_time
        real :: MM,x1,x2,x3,H,dE,T,dE2
        real :: R
        real ::  R_i, P_i, H_i, field(N,N,Nz),sss(N,N,Nz),sss_p(N,N,Nz)

        character(len=512) :: cFile, cFile2


        character*8 :: date
        character*10 :: time
        character*5 :: zone
        integer*4 :: values(8)


        ! melting 
        ! do T=6.0,6.1,0.2
        T=6.0
            s=0
            do ii=1,N
              do jj=1,N
                 call random_number(x1)
                 if(x1>0.5)then
                    s(ii,jj,1)=1
                 else
                    s(ii,jj,1)=-1
                 endif
              enddo
            enddo
            do ii=1,N
              do jj=1,N
                 call random_number(x1)
                 if(x1>0.5)then
                    s_p(ii,jj,1)=0
                 else
                    s_p(ii,jj,1)=-0
                 endif
              enddo
            enddo
            
            
            
            M=0
            do ii=1,N
                do jj=1,N
                    do kk=1,Nz
                        M=M+s(ii,jj,kk)
                    enddo
                enddo
            enddo
            
            
            Z=0 
            ! if(T>3.)then
                    anneal_steps=20000000
            ! else
            !         anneal_steps=2000000000
            ! endif
            do while(Z<anneal_steps)                     
                    Z=Z+1                                  
                    call random_number (x1)                
                    call random_number (x2)                
                    call random_number (x3)                
                    k1=x1*N+1; k2=x2*N+1;k3=1 
            
            
            
            
!                    if(mod(k1+k2,2)>0)then                
                    ! call random_number(x1)
                    ! if(x1>0.5)then
                            dE=0                                  
                            I=k1-1; if(I>0)then
                                    dE=dE+s(I,k2,k3)
                                    else
                                    dE=dE+s(N,k2,k3)
                                    endif
                            I=k1+1; if(I<N+1)then
                                    dE=dE+s(I,k2,k3)
                                    else
                                    dE=dE+s(1,k2,k3)
                                    endif
                            I=k2-1; if(I>0)then
                                    dE=dE+s(k1,I,k3)
                                    else
                                    dE=dE+s(k1,N,k3)
                                    endif
                            I=k2+1; if(I<N+1)then
                                    dE=dE+s(k1,I,k3)
                                    else
                                    dE=dE+s(k1,1,k3)
                                    endif
                            
                            dE2=0
                            I=k1-1; I2=k2-1
                                    if(I>0.and.I2>0.)then
                                            dE2=dE2+s(I,I2,k3)
                                    else if(I2>0) then
                                            dE2=dE2+s(N,I2,k3)
                                    else if (I>0) then
                                            dE2=dE2+s(I,N,k3)
                                    else
                                            dE2=dE2+s(N,N,k3)
                                    endif
                            I=k1+1; I2=k2-1
                                    if(I<N+1.and.I2>0)then
                                            dE2=dE2+s(I,I2,k3)
                                    else if (I2>0)then
                                            dE2=dE2+s(1,I2,k3)
                                    else if (I<N+1)then
                                            dE2=dE2+s(I,N,k3)
                                    else
                                            dE2=dE2+s(1,N,k3)
                                    endif
                            I=k1-1; I2=k2+1
                                    if(I>0.and.I2<N+1)then
                                            dE2=dE2+s(I,I2,k3)
                                    else if (I2<N+1)then
                                            dE2=dE2+s(N,I2,k3)
                                    else if (I>0)then
                                            dE2=dE2+s(I,1,k3)
                                    else
                                            dE2=dE2+s(N,1,k3)
                                    endif
                            I=k2+1; I2=k2+1
                                    if(I<N+1.and.I2<N+1)then
                                            dE2=dE2+s(I,I2,k3)
                                    else if (I2<N+1)then
                                            dE2=dE2+s(1,I2,k3)
                                    else if (I<N+1)then
                                            dE2=dE2+s(I,1,k3)
                                    else
                                            dE2=dE2+s(1,1,k3)
                                    endif
            
                            R=exp((-dE*J*s(k1,k2,k3)*2-dE2*J2*s(k1,k2,k3)-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2)/T)           
                            call random_number (x3)                
                            if( R>1.OR.x3<R )then
                                M=M-2*s(k1,k2,k3) 
                                s(k1,k2,k3) = -s(k1,k2,k3)                    
                            end if
                    ! else
                    !         R=exp(-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2/T)           
                    !         call random_number (x3)                
                    !         if( R>1.OR.x3<R )then
                    !             s_p(k1,k2,k3) = -s_p(k1,k2,k3)                    
                    !         end if
                    ! end if
            enddo

        ! enddo

        ! annealing
        do T=6.0,1.0,-0.2

        Z=0 
        if(T>3.)then
                anneal_steps=2000000
                decorrelation_time=10
        else
                anneal_steps=200000000
                decorrelation_time=1000
        endif
        do while(Z<anneal_steps)                     
                Z=Z+1                                  
                call random_number (x1)                
                call random_number (x2)                
                call random_number (x3)                
                k1=x1*N+1; k2=x2*N+1;k3=1 




!                if(mod(k1+k2,2)>0)then                
                ! call random_number(x1)
                ! if(x1>0.5)then
                        dE=0                                  
                        I=k1-1; if(I>0)then
                                dE=dE+s(I,k2,k3)
                                else
                                dE=dE+s(N,k2,k3)
                                endif
                        I=k1+1; if(I<N+1)then
                                dE=dE+s(I,k2,k3)
                                else
                                dE=dE+s(1,k2,k3)
                                endif
                        I=k2-1; if(I>0)then
                                dE=dE+s(k1,I,k3)
                                else
                                dE=dE+s(k1,N,k3)
                                endif
                        I=k2+1; if(I<N+1)then
                                dE=dE+s(k1,I,k3)
                                else
                                dE=dE+s(k1,1,k3)
                                endif
                        
                        dE2=0
                        I=k1-1; I2=k2-1
                                if(I>0.and.I2>0.)then
                                        dE2=dE2+s(I,I2,k3)
                                else if(I2>0) then
                                        dE2=dE2+s(N,I2,k3)
                                else if (I>0) then
                                        dE2=dE2+s(I,N,k3)
                                else
                                        dE2=dE2+s(N,N,k3)
                                endif
                        I=k1+1; I2=k2-1
                                if(I<N+1.and.I2>0)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2>0)then
                                        dE2=dE2+s(1,I2,k3)
                                else if (I<N+1)then
                                        dE2=dE2+s(I,N,k3)
                                else
                                        dE2=dE2+s(1,N,k3)
                                endif
                        I=k1-1; I2=k2+1
                                if(I>0.and.I2<N+1)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2<N+1)then
                                        dE2=dE2+s(N,I2,k3)
                                else if (I>0)then
                                        dE2=dE2+s(I,1,k3)
                                else
                                        dE2=dE2+s(N,1,k3)
                                endif
                        I=k2+1; I2=k2+1
                                if(I<N+1.and.I2<N+1)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2<N+1)then
                                        dE2=dE2+s(1,I2,k3)
                                else if (I<N+1)then
                                        dE2=dE2+s(I,1,k3)
                                else
                                        dE2=dE2+s(1,1,k3)
                                endif

                        R=exp((-dE*J*s(k1,k2,k3)*2-dE2*J2*s(k1,k2,k3)-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2)/T)           
                        call random_number (x3)                
                        if( R>1.OR.x3<R )then
                            M=M-2*s(k1,k2,k3) 
                            s(k1,k2,k3) = -s(k1,k2,k3)                    
                        end if
                ! else
                !         R=exp(-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2/T)           
                !         call random_number (x3)                
                !         if( R>1.OR.x3<R )then
                !             s_p(k1,k2,k3) = -s_p(k1,k2,k3)                    
                !         end if
                ! end if
        enddo

        A=0
        Z=0
        ss=0
        ss_p=0
        write(cFile,"(f5.2)")T
        write(cFile2,"(f5.2)")T
        cFile = 'M'//Trim(AdjustL(cFile))//'.txt'
        cFile2 = 'S'//Trim(AdjustL(cFile2))//'.txt'
        
        open(10,file=cFile)
        open(12,file=cFile2)
        do  while(Z<anneal_steps)
                ! call random_number(x1)
                ! if(x1>0.5)then
                        call random_number (x1)                
                        call random_number (x2)                
                        call random_number (x3)                
                        k1=x1*N+1; k2=x2*N+1;k3=1 
                        dE=0                                  
                        I=k1-1; if(I>0)then
                                dE=dE+s(I,k2,k3)
                                else
                                dE=dE+s(N,k2,k3)
                                endif
                        I=k1+1; if(I<N+1)then
                                dE=dE+s(I,k2,k3)
                                else
                                dE=dE+s(1,k2,k3)
                                endif
                        I=k2-1; if(I>0)then
                                dE=dE+s(k1,I,k3)
                                else
                                dE=dE+s(k1,N,k3)
                                endif
                        I=k2+1; if(I<N+1)then
                                dE=dE+s(k1,I,k3)
                                else
                                dE=dE+s(k1,1,k3)
                                endif
                        
                        dE2=0
                        I=k1-1; I2=k2-1
                                if(I>0.and.I2>0.)then
                                        dE2=dE2+s(I,I2,k3)
                                else if(I2>0) then
                                        dE2=dE2+s(N,I2,k3)
                                else if (I>0) then
                                        dE2=dE2+s(I,N,k3)
                                else
                                        dE2=dE2+s(N,N,k3)
                                endif
                        I=k1+1; I2=k2-1
                                if(I<N+1.and.I2>0)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2>0)then
                                        dE2=dE2+s(1,I2,k3)
                                else if (I<N+1)then
                                        dE2=dE2+s(I,N,k3)
                                else
                                        dE2=dE2+s(1,N,k3)
                                endif
                        I=k1-1; I2=k2+1
                                if(I>0.and.I2<N+1)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2<N+1)then
                                        dE2=dE2+s(N,I2,k3)
                                else if (I>0)then
                                        dE2=dE2+s(I,1,k3)
                                else
                                        dE2=dE2+s(N,1,k3)
                                endif
                        I=k2+1; I2=k2+1
                                if(I<N+1.and.I2<N+1)then
                                        dE2=dE2+s(I,I2,k3)
                                else if (I2<N+1)then
                                        dE2=dE2+s(1,I2,k3)
                                else if (I<N+1)then
                                        dE2=dE2+s(I,1,k3)
                                else
                                        dE2=dE2+s(1,1,k3)
                                endif

                        R=exp((-dE*J*s(k1,k2,k3)*2-dE2*J2*s(k1,k2,k3)-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2)/T)        
                        call random_number (x3)                
                        if( R>1.OR.x3<R )then
                            M=M-2*s(k1,k2,k3) 
                            s(k1,k2,k3) = -s(k1,k2,k3)                    
                        end if
                        Z=Z+1
                        A=A+M
                        if (mod(Z,decorrelation_time) == 0) then
                            write(10,"(i5)") M
                            do kk=1,Nz 
                            do jj=1,N 
                                    write(12,201) s(:,jj,kk)
                            enddo
                            enddo
                            do kk=1,Nz 
                            do jj=1,N 
                                    write(12,201) -s(:,jj,kk)
                            enddo
                            enddo
                        endif
                        ss=ss+s
                        ss_p=ss_p+s_p
                        !if(mod(Z,10000)==0)then
                        !print*,ss
                        !endif
                ! else
                !         R=exp(-s_p(k1,k2,k3)*J_p*s(k1,k2,k3)*2/T)           
                !         call random_number (x3)                
                !         if( R>1.OR.x3<R )then
                !             s_p(k1,k2,k3) = -s_p(k1,k2,k3)                    
                !         end if
                ! end if
        enddo



        MM=A/(Z*N*N*Nz)
        sss=ss/real(Z)        
        sss_p=ss_p/real(Z)        
        !print*,"T=",T,"M=",MM
        call date_and_time(date,time,zone,values)
        write(cFile,"(f5.2)")T
        cFile = ''//Trim(AdjustL(cFile))//'.txt'
        open(11,file=cFile)
        do kk=1,Nz 
        do jj=1,N 
        !do ii=1,N 
            !print*,sss(:,jj,kk)
            write(11,200)sss(:,jj,kk)!-sss_p(:,jj,kk)
        !enddo
        enddo
        enddo 
200 format(36(f10.5))
201 format(36(i3))
        enddo
        end                  
