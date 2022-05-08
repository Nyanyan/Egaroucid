window.onload = function() {
    //Language
    var language = (window.navigator.languages && window.navigator.languages[0]) ||
    window.navigator.language ||
    window.navigator.userLanguage ||
    window.navigator.browserLanguage;

    console.log(language);

    if (language == 'ja')
        window.location.href = "https://www.egaroucid.nyanyan.dev/ja";
    else
        window.location.href = "https://www.egaroucid.nyanyan.dev/en";
}