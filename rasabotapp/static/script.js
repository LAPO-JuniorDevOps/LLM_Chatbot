let slideIndex = 0;

function showSlides() {
    let slides = document.getElementsByClassName("mySlides");
    
    for (let i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    slideIndex++;
    if (slideIndex > slides.length) {slideIndex = 1}

    slides[slideIndex - 1].style.display = "inline-block";
    setTimeout(showSlides, 2000);
}


showSlides();

let writeIndex = 0;

function showwrites() {
    let write = document.getElementsByClassName("mywritings");
    
    for (let i = 0; i < write.length; i++) {
        write[i].style.display = "none";
    }

    writeIndex++;
    if (writeIndex > write.length) {writeIndex = 1}

    write[writeIndex - 1].style.display = "inline-block";
    setTimeout(showwrites, 2000);
}


showwrites();


window.addEventListener('scroll', changeHeaderStyle);

let header = document.querySelector('.death');
let head = document.querySelector('.bath');
let hd = document.querySelector('.bath1');
let originalContent = header.innerHTML;
let scrollThreshold = 3;
let lastScroll = 0;

function changeHeaderStyle() {
    let currentScroll = window.scrollY;
    let ds1Element = document.querySelector('.DS1');
    let ds3Element = document.querySelector('.nig');
    let ds4Element = document.querySelector('.bath1');
    
    if (currentScroll > lastScroll && currentScroll > scrollThreshold) {
        if (ds4Element) {
            ds4Element.style.visibility = 'visible';
        }
        if (ds3Element) {
            ds3Element.style.visibility = 'visible';
        }
        if (ds1Element) {
            ds1Element.style.visibility = 'hidden';
        }
    } else if (currentScroll < lastScroll) {
        if (ds4Element) {
            ds4Element.style.visibility = 'visible';
        }
        if (ds3Element) {
            ds3Element.style.visibility = 'hidden';
        }
        if (ds1Element) {
            ds1Element.style.visibility = 'visible';
        }
    }

    lastScroll = currentScroll;

    if (currentScroll > scrollThreshold) {
        header.innerHTML = '';
        head.innerHTML = ds1Element.innerHTML;
        hd.innerHTML = ds4Element.innerHTML;
    } else {
        header.innerHTML = originalContent;
        head.innerHTML = '';
    }
}
