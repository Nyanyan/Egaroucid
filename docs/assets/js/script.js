const lang_lst = ['ja', 'en']
const len_lang = 2;

function reset_lang(){
    for (let i = 0; i < len_lang; ++i){
        document.getElementById(lang_lst[i]).hidden = true;
    }
}

function ja(){
    reset_lang();
    document.getElementById('ja').hidden = false;
}

function en(){
    reset_lang();
    document.getElementById('en').hidden = false;
}