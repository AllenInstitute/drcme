      subroutine delcol(r,p,k,z,n,nz)
      integer p,k,n,nz
      double precision r(p,*), z(n,nz)
      integer p1,i,j
      double precision a,b,c,s,tau
      p1 = p-1
      i = k
23000 if(.not.(i .lt. p))goto 23002
      a = r(i,i)
      b = r(i+1,i)
      if(.not.(b .eq. 0d0))goto 23003
      goto 23001
23003 continue
      if(.not.(dabs(b) .gt. dabs(a)))goto 23005
      tau = -a/b
      s = 1/dsqrt(1d0+tau*tau)
      c = s * tau 
      goto 23006
23005 continue
      tau = -b/a
      c = 1/dsqrt(1d0+tau*tau)
      s = c * tau 
23006 continue
      r(i,i) = c*a - s*b
      r(i+1,i) = s*a + c*b
      j = i +1
23007 if(.not.(j .le. p1))goto 23009
      a = r(i,j)
      b = r(i+1,j)
      r(i,j) = c*a - s *b
      r(i+1,j) = s*a + c * b
       j = j+1
      goto 23007
23009 continue
      j=1
23010 if(.not.(j .le. nz))goto 23012
      a = z(i,j)
      b = z(i+1,j)
      z(i,j) = c*a - s*b
      z(i+1,j) = s*a + c*b
       j = j+1
      goto 23010
23012 continue
23001  i=i+1
      goto 23000
23002 continue
      return
      end
