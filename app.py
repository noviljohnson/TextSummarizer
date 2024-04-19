import streamlit_authenticator as stauth
import streamlit as st
import requests

summarizer_url = "http://127.0.0.1:5000"

if "login_status" not in st.session_state:
    st.session_state['login_status'] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("------- Text Summarizer ------")
    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Login":
        st.subheader("Login Section")

        st.session_state["email"] = st.text_input("Email")
        st.session_state["password"] = st.text_input("Password",type='password')

        if st.checkbox("Login"):
            user_details = {"email":st.session_state["email"],"password": st.session_state["password"]}
            res = requests.get(summarizer_url+'/login', json=user_details)

            if res.status_code == 200:
                log = res.json()['log']
                if log == True:
                    st.success(f"Logged In as {st.session_state['email']}")
                    st.session_state['login_status'] = True
                elif log == 'No Details Found':
                    st.write('Please Sign up')
                else:
                    st.warning("Incorrect Email/Password")
            else:
                st.warning("Try Reloading the page and relogin")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        email = st.text_input("Email")
        new_password = st.text_input("Password",type='password')
        full_name = st.text_input('Full Name')

        if st.button("Signup"):
            user_details = {"email":email, "new_password":new_password,"full_name":full_name}
            res = requests.get(summarizer_url+'/update_user', json=user_details)
            if res.status_code == 200 and res.json()['log'] != 'Exists':
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")
            elif res.status_code == 200 and res.json()['log'] == 'Exists':
                st.write('User Already Exists. Please Login')

    if st.session_state['login_status'] == True:
        st.markdown("<h3 style='color:#A9A9A9;'>Chat Session</h3>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(""):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                user_input = {'email':st.session_state["email"], 'text':prompt}
                res = requests.get(summarizer_url+'/summarize', json=user_input)

                if res.status_code == 200:

                    if 'answer' in res.json().keys():
                        st.session_state["assistant_message"] = res.json()['answer']
                        st.write(res.json()['answer'])
                    else:
                        st.session_state["assistant_message"] = res.json()['summary']
                        st.write(st.session_state["assistant_message"])
                else:
                    st.write("Please Try again")
                    st.session_state["assistant_message"] = ""
                    print(res)
                
                st.session_state.messages.append({"role": "assistant", "content": st.session_state["assistant_message"]})



if __name__ == '__main__':
	main()