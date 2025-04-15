from difflib import SequenceMatcher


def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


if __name__ == "__main__":
    str1 = "hello world"
    str2 = "hello there"
    print(string_similarity(str1, str2))

    # calculate code fragment similarity use cosine similarity
    code1 = """
#include <stdio.h>  
int main() {
    printf("Hello, World!\n");
    return 0;
}
"""
    code2 = """
    #include <stdio.h>
    int main() {
        sv -> printf("Hello, World!\n");
        return 0;
    }
    """

    print(string_similarity(code1, code2))

    code3 = """
#include <stdio.h>
int main() {
    printf("Hello, World!\n");
    return 0;
}
"""

    code4 = """
#include <stdio.h>
int main() {
    std::cout << "Hello, World!\n";
    return 0;
}
"""

    print(string_similarity(code3, code4))
