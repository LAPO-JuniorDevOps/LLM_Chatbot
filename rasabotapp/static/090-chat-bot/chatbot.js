function toggleNav() {
    let navBar = document.getElementById("navBar");
    navBar.classList.toggle("show-nav");
}

const navBar = document.getElementById('navBar');
  const animation = document.querySelector('.nav-animation');

  navBar.addEventListener('scroll', () => {
    if (navBar.scrollTop > 10) {
      animation.style.height = '60px';
    } else {
      animation.style.height = '140px';
    }
  });

  const ddd = document.getElementsByClassName('mmm');
  let findBall = ddd[0];
  
  for (let i = 0; i < ddd.length; i++) {
    ddd[i].addEventListener('scroll', () => {
      if (ddd[i].scrollTop > 50) {
        findBall.style.color = '#1973b8';
      } else {
        findBall.style.color = '';
      }
    });
  }
  